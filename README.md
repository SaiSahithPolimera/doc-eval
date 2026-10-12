# doc-eval

**doc-eval** is a prototype for evaluating document-AI extraction pipelines using English invoice images. It combines OCR, local LLM-based structured extraction, reference-based metrics, and batch reports to help identify model errors and compare future prompt/model versions.

> **Project status:** Prototype. The five-invoice batch pipeline has run successfully, but the current scores are preliminary and should not be treated as representative model benchmarks. Continuous evaluation, automated CI quality gates, and systematic experiment tracking are planned extensions.

## Overview

doc-eval evaluates this pipeline:

```text
Voxel51 English invoice dataset
              |
              v
       Invoice image
              |
              v
          PaddleOCR
              |
              v
     Recognized OCR text
              |
              v
   Qwen model through Ollama
              |
              v
     Structured invoice JSON
              |
              v
 Compare with ground-truth JSON
              |
              v
   Metrics + JSON/CSV reports
```

## Current features

- **Invoice dataset:** Downloads annotated samples from the [Voxel51 High Quality Invoice Images for OCR dataset](https://huggingface.co/datasets/Voxel51/high-quality-invoice-images-for-ocr) and exports images, annotations, and a manifest locally.
- **OCR:** Uses PaddleOCR to recognize invoice text. Raw OCR results can be retained as JSON, while extracted text is cached as `.txt` for the extraction model.
- **Structured extraction:** Uses a locally hosted Qwen model through Ollama to convert OCR text into the dataset's nested invoice schema.
- **Reference-based evaluation:** Compares predicted JSON with ground-truth JSON and reports field accuracy, exact-match rate, numeric accuracy, completeness, and per-field accuracy.
- **Batch evaluation:** Processes multiple invoice samples and saves aggregate results to JSON and per-invoice results to CSV.
- **Testing:** Pytest tests cover core metric behavior, including numeric formatting normalization.
- **Observability exploration:** Arize Phoenix has been set up, and an evaluation trace has been observed. Full end-to-end tracing of OCR, model inference, and evaluation remains work in progress.

## Technology stack

| Component | Technology | Purpose |
|---|---|---|
| Dataset | Voxel51 / Hugging Face | English invoice images and annotations |
| OCR | PaddleOCR / PaddlePaddle | Text detection and recognition |
| Extraction model | Qwen via Ollama | Convert OCR text to structured JSON |
| Evaluation | Python | Compare predictions against references |
| Tests | pytest | Test metric behavior |
| Observability | Arize Phoenix | Trace inspection and evaluation observability |
| Version control | Git | Track source code and prompt changes |

## Repository layout

The layout below represents the intended/current working structure; filenames may differ slightly as the project evolves.

```text
doc-eval/
├── data/
│   ├── invoices/
│   │   └── english/
│   │       ├── images/
│   │       ├── ground_truth/
│   │       ├── reference_ocr/
│   │       └── manifest.json
│   ├── ocr_outputs/
│   └── predictions/
├── reports/
│   ├── batch_eval_summary.json
│   └── batch_eval_results.csv
├── src/
│   ├── evaluation/
│   │   ├── batch_eval.py
│   │   └── metrics.py
│   └── extraction/
│       └── english_extractor.py
├── tests/
│   └── test_metrics.py
├── requirements.txt
└── README.md
```

Generated data and reports may be omitted from Git depending on repository `.gitignore` policy. Check the actual project tree before relying on exact paths.

## Prerequisites

- Python version compatible with the installed PaddleOCR/PaddlePaddle packages. This prototype was developed in a Windows virtual environment with Python 3.13; compatibility depends on the package versions and platform.
- Git
- Ollama installed and running
- A Qwen model pulled in Ollama
- Internet access for downloading the dataset and model files

## Setup

### 1. Clone the repository

```powershell
git clone <YOUR_REPOSITORY_URL>
cd doc-eval
```

If you already have the project locally, open PowerShell at the project root instead.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, consult your system's execution-policy settings or activate the environment using your preferred terminal.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If you are setting up the project from scratch, verify the versions in `requirements.txt` against the official PaddleOCR/PaddlePaddle installation guidance for your platform. Avoid changing working package versions without checking compatibility.

### 4. Set up Ollama and Qwen

Install Ollama from [ollama.com](https://ollama.com/), start it, and pull the model tag you intend to use. For example, if available in your local Ollama installation:

```powershell
ollama pull qwen3:4b-instruct
ollama list
```

Use the exact installed model tag when running extraction. The example tag may differ from the one available in your environment.

### 5. Download the English invoice dataset

The downloader used in this project is `src.evaluation.downloader`. Confirm that this module exists in your checkout, then run:

```powershell
python -m src.evaluation.downloader --limit 300
```

The downloader exports available annotated image/JSON pairs into `data/invoices/english/`. It skips samples without usable annotations. The number exported can be lower than the requested sample limit.

## Run the pipeline

Run commands from the repository root.

### 1. Run OCR on one invoice

The OCR runner's filename and command-line arguments may depend on your current implementation. If `src/evaluation/run_ocr.py` exists and accepts an image path, an example is:

```powershell
python src\evaluation\run_ocr.py "data\invoices\english\images\english_0000.jpg"
```

Inspect the resulting text file and raw OCR JSON under `data/ocr_outputs/`. Check the actual output filenames generated by your runner.

### 2. Extract structured fields with Qwen

If the English extractor is available as `src.extraction.english_extractor`:

```powershell
python -m src.extraction.english_extractor `
  --ocr "data\ocr_outputs\english_0000_page_1.txt" `
  --output "data\predictions\english_0000.json"
```

The extractor expects OCR text and returns a JSON object using the Voxel51 annotation structure.

### 3. Evaluate one invoice

```powershell
python -m src.evaluation.metrics `
  "data\invoices\english\ground_truth\english_0000.json" `
  "data\predictions\english_0000.json"
```

The metric CLI should print the comparison report for that prediction/reference pair.

### 4. Run batch evaluation

```powershell
python -m src.evaluation.batch_eval --limit 5
```

Increase the sample count after verifying the initial run:

```powershell
python -m src.evaluation.batch_eval --limit 20
```

The batch evaluator is expected to write:

- `reports/batch_eval_summary.json` — aggregate metrics and per-field accuracy.
- `reports/batch_eval_results.csv` — per-invoice results and processing status.

If your local implementation uses different module names or arguments, use `python -m <module> --help` and adjust these commands to match the code.

## Evaluation metrics

- **Field accuracy:** Fraction of compared reference fields that match the prediction.
- **Exact-match rate:** Fraction of invoices whose evaluated fields all match.
- **Numeric accuracy:** Accuracy for selected numeric fields after parsing supported numeric formats.
- **Field completeness:** Measures whether populated reference fields are populated in predictions.
- **Per-field accuracy:** Shows which schema fields are most frequently correct or incorrect.

Metrics are only meaningful when the prediction and ground-truth schemas match and normalization rules are appropriate. The current prototype needs additional validation for line-item grouping/order, empty-reference fields, address normalization, date formats, and false-positive extraction. Numeric parsing must be tested carefully with decimal commas and thousands separators.

## Testing

Run the metric tests:

```powershell
python -m pytest tests/test_metrics.py -v
```

In the development environment, pytest's automatic plugin loading previously encountered a compatibility error while importing Phoenix and Strawberry. Disabling plugin autoload allowed the metric tests to run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest tests/test_metrics.py -v
```

This is a workaround for test startup, not a fix for Phoenix's dependency compatibility. If normal pytest startup fails, check the installed Phoenix and Strawberry versions and resolve their compatibility before relying on plugin autoload.

## Current benchmark status

A five-invoice batch has completed successfully. One reported run produced approximately:

- Aggregate field accuracy: **63.7%**
- Numeric accuracy: **58.7%**
- Exact-match rate: **0%**
- Successful invoices: **5/5**

## References

- [Voxel51 invoice OCR dataset](https://huggingface.co/datasets/Voxel51/high-quality-invoice-images-for-ocr)
- [PaddleOCR documentation](https://www.paddleocr.ai/)
- [Ollama](https://ollama.com/)
- [Arize Phoenix](https://phoenix.arize.com/)
- [DeepEval](https://docs.confident-ai.com/)
- [LangSmith evaluation documentation](https://docs.smith.langchain.com/evaluation)

---

**doc-eval is a learning-oriented prototype.** Its goal is not only to extract invoice data, but to build a reproducible workflow for measuring, debugging, and comparing document-AI model behavior.
