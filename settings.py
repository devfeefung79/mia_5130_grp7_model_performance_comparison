import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / '.env')

REGION = 'us-east-2'
BEARER_TOKEN = os.getenv('AWS_BEARER_TOKEN_BEDROCK')

GENERATION_MODELS = {
	'nova_lite': 'amazon.nova-lite-v1:0',
	'nvidia_nemotron_nano_12b': 'nvidia.nemotron-nano-12b-v2',
	'qwen3_32b': 'qwen.qwen3-32b-v1:0',
	'llama_scout_17b': 'us.meta.llama4-scout-17b-instruct-v1:0',
}

generation_models = list(GENERATION_MODELS.keys())

LLM_JUDGE_MODEL = 'nova_lite'

WORKLOADS = [
	'product_catalogue_extraction',
	'customer_issue_classification',
	'product_review_summarization',
	'customer_response_drafting',
]

INPUT_DIR = PROJECT_ROOT / 'data' / 'outputs'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'outputs'
CRITERIA_DIR = PROJECT_ROOT / 'evaluation_criteria'
