# AWS Bedrock Model Performance Comparison

## Setup

Install `uv`, create a virtual environment, and install the dependencies:

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Configure Amazon Bedrock

1. In the AWS Console, select the `us-east-2` region, open **Amazon Bedrock > API keys**, and generate a **short-term API key**. It expires when your console session ends, up to 12 hours.
2. Copy `.env.example` to `.env` and set `AWS_BEARER_TOKEN_BEDROCK` to the generated key:

   ```bash
   cp .env.example .env
   ```

   Keep the key private; do not commit `.env`.

See the [Amazon Bedrock API key guide](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html) for details.

## Run

Run the workload scripts for the datasets you want to process:

```bash
python scripts/product_catalogue_extraction.py
python scripts/customer_issue_classification.py
python scripts/product_review_summarization.py
python scripts/customer_response_drafting.py
```

Then evaluate and compare the results:

```bash
python scripts/llm_judge.py
python scripts/evaluate_results.py
```

Results are saved under `data/outputs/`. Model requests may incur AWS charges.
