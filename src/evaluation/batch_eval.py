
import argparse
import csv
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from paddleocr import PaddleOCR

from src.evaluation.metrics import evaluate, parse_decimal
from src.extraction.extractor import (
    DEFAULT_MODEL,
    extract_invoice,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "invoices" / "english"

IMAGE_DIR = DATA_DIR / "images"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
OCR_DIR = ROOT / "data" / "ocr_outputs"
PREDICTION_DIR = ROOT / "data" / "predictions"
REPORT_DIR = ROOT / "reports"

NUMERIC_FIELDS = {
    "items[*].quantity",
    "items[*].total_price",
    "subtotal.tax",
    "subtotal.discount",
    "subtotal.total",
}

_ocr = None


def find_image(invoice_id: str) -> Path:
    """Find an invoice image by its ID."""
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        path = IMAGE_DIR / f"{invoice_id}{extension}"
        if path.is_file():
            return path

    raise FileNotFoundError(f"Image not found for {invoice_id}")


def get_ocr_text(invoice_id: str, image_path: Path) -> str:
    """Reuse cached OCR text or run PaddleOCR."""
    global _ocr

    text_path = OCR_DIR / f"{invoice_id}_page_1.txt"

    if text_path.is_file():
        cached_text = text_path.read_text(encoding="utf-8")
        if cached_text.strip():
            return cached_text

    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    results = _ocr.predict(str(image_path))
    all_lines = []

    for page_number, result in enumerate(results, start=1):
        raw = result.json
        payload = raw.get("res", raw)

        raw_path = OCR_DIR / (
            f"{invoice_id}_page_{page_number}_res.json"
        )
        OCR_DIR.mkdir(parents=True, exist_ok=True)

        raw_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        all_lines.extend(payload.get("rec_texts", []))

    text = "\n".join(
        str(line) for line in all_lines if str(line).strip()
    )

    if not text.strip():
        raise ValueError(f"No recognized text for {invoice_id}")

    text_path.write_text(text, encoding="utf-8")
    return text


def normalize_field_path(path: str) -> str:
    """Group line-item metrics across different array indexes."""
    return re.sub(r"items\[\d+\]", "items[*]", path)


def parse_number(value):
    return parse_decimal(value)

def is_nonempty(value) -> bool:
    return value is not None and str(value).strip() != ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")

    truth_files = sorted(GROUND_TRUTH_DIR.glob("*.json"))

    if not truth_files:
        raise FileNotFoundError(
            f"No ground-truth JSON files found in {GROUND_TRUTH_DIR}"
        )

    selected = []
    for truth_path in truth_files:
        try:
            image_path = find_image(truth_path.stem)
            selected.append((truth_path, image_path))
        except FileNotFoundError:
            continue

    if not selected:
        raise FileNotFoundError("No matching image/annotation pairs found.")

    selected = selected[:args.limit]

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    field_correct = defaultdict(int)
    field_counts = defaultdict(int)

    numeric_correct = 0
    numeric_count = 0
    populated_reference_count = 0
    populated_prediction_count = 0
    populated_correct_count = 0
    total_correct_fields = 0
    total_reference_fields = 0

    for index, (truth_path, image_path) in enumerate(selected, start=1):
        invoice_id = truth_path.stem

        print(f"\n[{index}/{len(selected)}] {invoice_id}")

        try:
            reference = json.loads(
                truth_path.read_text(encoding="utf-8")
            )

            ocr_text = get_ocr_text(invoice_id, image_path)

            prediction = extract_invoice(ocr_text, args.model)

            prediction_path = PREDICTION_DIR / f"{invoice_id}.json"
            prediction_path.write_text(
                json.dumps(prediction, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            report = evaluate(reference, prediction)

            total_correct_fields += report["correct_fields"]
            total_reference_fields += report["total_reference_fields"]

            for path, result in report["field_results"].items():
                field = normalize_field_path(path)
                ref_value = result["reference"]
                pred_value = result["prediction"]

                field_counts[field] += 1
                field_correct[field] += int(result["correct"])

                if is_nonempty(ref_value):
                    populated_reference_count += 1
                    if is_nonempty(pred_value):
                        populated_prediction_count += 1
                    if result["correct"]:
                        populated_correct_count += 1

                if field in NUMERIC_FIELDS:
                    ref_number = parse_number(ref_value)

                    if ref_number is not None:
                        numeric_count += 1
                        pred_number = parse_number(pred_value)

                        if (
                            pred_number is not None
                            and pred_number == ref_number
                        ):
                            numeric_correct += 1

            rows.append({
                "invoice_id": invoice_id,
                "field_accuracy": report["field_accuracy"],
                "exact_match": report["exact_match"],
                "correct_fields": report["correct_fields"],
                "total_fields": report["total_reference_fields"],
                "status": "success",
            })

            print(f"Field accuracy: {report['field_accuracy']:.2%}")
            print(f"Exact match: {report['exact_match']}")

        except Exception as exc:
            print(f"FAILED: {exc}")

            rows.append({
                "invoice_id": invoice_id,
                "field_accuracy": "",
                "exact_match": "",
                "correct_fields": "",
                "total_fields": "",
                "status": f"failed: {exc}",
            })

    successful = [row for row in rows if row["status"] == "success"]

    if not successful:
        raise RuntimeError("Every invoice failed. See errors above.")

    summary = {
        "dataset": "Voxel51/high-quality-invoice-images-for-ocr",
        "model": args.model,
        "requested_invoices": len(selected),
        "successful_invoices": len(successful),
        "failed_invoices": len(rows) - len(successful),
        "mean_invoice_field_accuracy": sum(
            float(row["field_accuracy"]) for row in successful
        ) / len(successful),
        "aggregate_field_accuracy": (
            total_correct_fields / total_reference_fields
            if total_reference_fields else None
        ),
        "exact_match_rate": sum(
            bool(row["exact_match"]) for row in successful
        ) / len(successful),
        "numeric_accuracy": (
            numeric_correct / numeric_count if numeric_count else None
        ),
        "field_completeness": (
            populated_prediction_count / populated_reference_count
            if populated_reference_count else None
        ),
        "per_field_accuracy": {
            field: field_correct[field] / count
            for field, count in sorted(field_counts.items())
        },
    }

    summary_path = REPORT_DIR / "batch_eval_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = REPORT_DIR / "batch_eval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== BATCH EVALUATION SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
