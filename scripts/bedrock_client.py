import sys
import time
from pathlib import Path

import boto3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from settings import BEARER_TOKEN, GENERATION_MODELS, REGION


if not BEARER_TOKEN:
	raise ValueError('AWS_BEARER_TOKEN_BEDROCK not found in .env')

bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)

# Backward-compatible alias used by older imports.
MODELS = GENERATION_MODELS
generation_models = GENERATION_MODELS


def invoke_model(model_name, prompt, max_tokens=1000, temperature=0.0):

	if model_name not in GENERATION_MODELS:
		raise ValueError(f'Unknown model: {model_name}')

	model_id = GENERATION_MODELS[model_name]
	start_time = time.perf_counter()

	try:
		response = bedrock_runtime.converse(
			modelId=model_id,
			messages=[{'role': 'user', 'content': [{'text': prompt}]}],
			inferenceConfig={'maxTokens': max_tokens, 'temperature': temperature},
		)

		latency = time.perf_counter() - start_time

		output = ''.join(
			block['text'] for block in response['output']['message']['content'] if 'text' in block
		)

		usage = response.get('usage', {})

		input_tokens = usage.get('inputTokens', 0)
		output_tokens = usage.get('outputTokens', 0)

		return {
			'model_name': model_name,
			'model_id': model_id,
			'output': output,
			'input_tokens': input_tokens,
			'output_tokens': output_tokens,
			'total_tokens': input_tokens + output_tokens,
			'latency_seconds': round(latency, 4),
			'status': 'success',
			'error': None,
		}

	except Exception as e:
		latency = time.perf_counter() - start_time

		return {
			'model_name': model_name,
			'model_id': model_id,
			'output': None,
			'input_tokens': 0,
			'output_tokens': 0,
			'total_tokens': 0,
			'latency_seconds': round(latency, 4),
			'status': 'failed',
			'error': str(e),
		}
