
import argparse
import json
from pathlib import Path

from paddleocr import PaddleOCR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an invoice image")
    parser.add_argument(
        "--output",
        default="data/ocr_outputs",
        help="Directory to save OCR outputs",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    results = ocr.predict(str(image_path))

    for page_number, result in enumerate(results, start=1):
        result.print()

        result.save_to_json(str(output_dir))
        result.save_to_img(str(output_dir))

        result_data = result.json
        payload = result_data.get("res", result_data)

        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])

        print(f"\n--- Page {page_number} ---")

        for index, text in enumerate(texts):
            score = (
                float(scores[index])
                if index < len(scores)
                else None
            )

            if score is None:
                print(text)
            else:
                print(f"{score:.3f} | {text}")

        text_path = output_dir / (
            f"{image_path.stem}_page_{page_number}.txt"
        )
        text_path.write_text(
            "\n".join(texts),
            encoding="utf-8",
        )

        print(f"\nRecognized text saved to: {text_path}")


if __name__ == "__main__":
    main()
