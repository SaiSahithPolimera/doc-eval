from src.evaluation.metrics import evaluate


def test_identical_json_scores_one():
    reference = {
        "invoice_number": "INV-001",
        "total": 1190.00,
    }

    report = evaluate(reference, reference)

    assert report["field_accuracy"] == 1.0
    assert report["exact_match"] is True


def test_wrong_total_is_detected():
    reference = {
        "invoice_number": "INV-001",
        "total": 1190.00,
    }

    prediction = {
        "invoice_number": "INV-001",
        "total": 1090.00,
    }

    report = evaluate(reference, prediction)

    assert report["field_accuracy"] == 0.5
    assert report["exact_match"] is False


def test_missing_field_is_detected():
    reference = {
        "invoice_number": "INV-001",
        "total": 1190.00,
    }

    prediction = {
        "invoice_number": "INV-001",
    }

    report = evaluate(reference, prediction)

    assert report["field_accuracy"] == 0.5
    assert "total" in report["missing_fields"]


def test_decimal_comma_and_decimal_point_are_equivalent():
    reference = {
        "items": [
            {
                "quantity": "1.00",
                "total_price": "46.55",
            }
        ],
        "subtotal": {
            "tax": "21.18",
            "discount": "",
            "total": "232.95",
        },
    }

    prediction = {
        "items": [
            {
                "quantity": "1,00",
                "total_price": "46,55",
            }
        ],
        "subtotal": {
            "tax": "21,18",
            "discount": "",
            "total": "232,95",
        },
    }

    report = evaluate(reference, prediction)

    assert report["field_accuracy"] == 1.0
    assert report["exact_match"] is True
