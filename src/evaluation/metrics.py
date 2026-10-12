
import json
import math
from datetime import datetime
import re
from decimal import Decimal, InvalidOperation
import sys
from numbers import Real
from pathlib import Path


def flatten(value, prefix=""):
    result = {}

    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))

        if not value and prefix:
            result[prefix] = {}

    elif isinstance(value, list):
        if not value and prefix:
            result[prefix] = []

        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            result.update(flatten(child, path))

    else:
        result[prefix] = value

    return result



def values_equal(reference, prediction):

    if reference is None or prediction is None:
        return reference is prediction

    if isinstance(reference, bool) or isinstance(prediction, bool):
        return reference == prediction

    if isinstance(reference, Real) and isinstance(prediction, Real):
        return math.isclose(
            float(reference),
            float(prediction),
            abs_tol=0.01,
            rel_tol=0.0,
        )

    if isinstance(reference, str) and isinstance(prediction, str):
        try:
            reference_dt = datetime.fromisoformat(
                reference.replace("Z", "+00:00")
            )
            prediction_dt = datetime.fromisoformat(
                prediction.replace("Z", "+00:00")
            )
            return reference_dt == prediction_dt
        except ValueError:
            pass

        normalize = lambda s: " ".join(s.casefold().split())
        return normalize(reference) == normalize(prediction)

    return reference == prediction

def evaluate(reference, prediction):
    expected = flatten(reference)
    actual = flatten(prediction)

    if not expected:
        raise ValueError("Ground truth contains no comparable fields.")

    correct = 0
    field_results = {}

    for path, expected_value in expected.items():
        predicted_value = actual.get(path)
        matches = (
            path in actual
            and field_values_equal(path, expected_value, actual.get(path))
        )

        correct += int(matches)
        field_results[path] = {
            "correct": matches,
            "reference": expected_value,
            "prediction": predicted_value,
        }

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))

    return {
        "field_accuracy": correct / len(expected),
        "correct_fields": correct,
        "total_reference_fields": len(expected),
        "missing_fields": missing,
        "extra_fields": extra,
        "exact_match": correct == len(expected) and not extra,
        "field_results": field_results,
    }


NUMERIC_FIELD_PATTERN = re.compile(
    r"^(?:items\[\d+\]\.(?:quantity|total_price)"
    r"|subtotal\.(?:tax|discount|total))$"
)


def parse_decimal(value):
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None

    text = str(value).strip()
    text = text.replace("\u00a0", "").replace(" ", "")
    text = re.sub(r"[^\d,.\-+]", "", text)

    if not text or text in {"-", "+", ".", ",", "-.", "+."}:
        return None

    if "," in text and "." in text:
        if text.rfind(".") > text.rfind(","):
            text = text.replace(",", "")
        else:
            text = text.replace(".", "").replace(",", ".")

    elif "," in text:
        parts = text.rsplit(",", 1)

        if text.count(",") == 1 and len(parts[1]) in (1, 2):
            text = parts[0] + "." + parts[1]
        else:
            text = text.replace(",", "")

    elif text.count(".") > 1:
        parts = text.split(".")
        if len(parts[-1]) in (1, 2):
            text = "".join(parts[:-1]) + "." + parts[-1]
        else:
            text = "".join(parts)

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def field_values_equal(path, reference, prediction):
    if NUMERIC_FIELD_PATTERN.fullmatch(path):
        ref_num = parse_decimal(reference)
        pred_num = parse_decimal(prediction)

        if ref_num is not None or pred_num is not None:
            return (
                ref_num is not None
                and pred_num is not None
                and ref_num == pred_num
            )

    return values_equal(reference, prediction)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python src/evaluation/metrics.py "
            "<ground_truth.json> <prediction.json>"
        )

    reference_path = Path(sys.argv[1])
    prediction_path = Path(sys.argv[2])

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))

    report = evaluate(reference, prediction)

    print(json.dumps(report, ensure_ascii=False, indent=2))

