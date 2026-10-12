
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python -m src.evaluation.json_to_text "
            "<input.json> <output.txt>"
        )
        raise SystemExit(1)

    json_path = Path(sys.argv[1])
    text_path = Path(sys.argv[2])

    if not json_path.is_file():
        raise FileNotFoundError(f"OCR JSON not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))

    result = data.get("res", data)

    texts = result.get("rec_texts")

    if texts is None:
        raise ValueError(
            f"'rec_texts' not found. Available keys: "
            f"{list(result.keys())}"
        )

    lines = [str(text) for text in texts if str(text).strip()]

    if not lines:
        raise ValueError(
            "No recognized text found in rec_texts."
        )

    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Recognized lines: {len(lines)}")
    print(f"Saved text to: {text_path}")
    print("\nPreview:")
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
