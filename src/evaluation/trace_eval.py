
import argparse
import json
from pathlib import Path

from opentelemetry import trace
from phoenix.otel import register

from src.evaluation.metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--invoice-id", required=True)
    args = parser.parse_args()

    reference = json.loads(
        Path(args.reference).read_text(encoding="utf-8")
    )
    prediction = json.loads(
        Path(args.prediction).read_text(encoding="utf-8")
    )

    register(
        project_name="doc-eval",
        auto_instrument=False,
    )

    tracer = trace.get_tracer("doc-eval.evaluation")

    with tracer.start_as_current_span("invoice_evaluation") as span:
        span.set_attribute("invoice.id", args.invoice_id)
        span.set_attribute("evaluation.type", "field_accuracy")

        report = evaluate(reference, prediction)

        span.set_attribute(
            "evaluation.field_accuracy",
            float(report["field_accuracy"]),
        )
        span.set_attribute(
            "evaluation.exact_match",
            bool(report["exact_match"]),
        )
        span.set_attribute(
            "evaluation.correct_fields",
            int(report["correct_fields"]),
        )
        span.set_attribute(
            "evaluation.total_fields",
            int(report["total_reference_fields"]),
        )
        span.set_attribute(
            "evaluation.missing_fields",
            json.dumps(report["missing_fields"]),
        )
        span.set_attribute(
            "evaluation.extra_fields",
            json.dumps(report["extra_fields"]),
        )

        print(json.dumps(report, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    main()
