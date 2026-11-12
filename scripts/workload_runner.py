import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from bedrock_client import invoke_model

from settings import PROJECT_ROOT as SETTINGS_ROOT
from settings import generation_models

# Ground-truth / labelling columns that must never be shown to the model
EXCLUDED_PROMPT_COLUMNS = {'expected_response_points', 'expected_category', 'difficulty'}


def run_workload(workload, input_file):

	prompt_file = SETTINGS_ROOT / 'prompts' / f'{workload}.txt'
	output_dir = SETTINGS_ROOT / 'data' / 'outputs'
	output_file = output_dir / f'{workload}_results.csv'

	# Load input data and prompt
	df = pd.read_csv(input_file)
	prompt_template = prompt_file.read_text(encoding='utf-8')

	output_dir.mkdir(parents=True, exist_ok=True)

	results = []

	for _, row in df.iterrows():
		input_data = row.to_dict()
		prompt_data = {k: v for k, v in input_data.items() if k not in EXCLUDED_PROMPT_COLUMNS}

		prompt = prompt_template.replace(
			'{product_data}', json.dumps(prompt_data, ensure_ascii=False, default=str)
		)

		for model in generation_models:
			print(f'Running {workload} | {model}')

			try:
				result = invoke_model(model_name=model, prompt=prompt)

				error_message = result.get('error')
				if result['status'] == 'failed':
					print(
						f'Request failed | {workload} | {model} | '
						f'{input_data.get("item_id")}: '
						f'{error_message or "No error details returned"}',
						file=sys.stderr,
					)

				results.append(
					{
						'workload': workload,
						'product_id': input_data.get('item_id'),
						'model_name': model,
						'input_data': json.dumps(input_data, ensure_ascii=False, default=str),
						'output': result['output'],
						'input_tokens': result['input_tokens'],
						'output_tokens': result['output_tokens'],
						'latency_seconds': result['latency_seconds'],
						'status': result['status'],
						'error': error_message,
					}
				)

			except Exception as error:
				print(
					f'Request failed | {workload} | {model} | {input_data.get("item_id")}: {error}',
					file=sys.stderr,
				)

				results.append(
					{
						'workload': workload,
						'product_id': input_data.get('item_id'),
						'model_name': model,
						'input_data': json.dumps(input_data, ensure_ascii=False, default=str),
						'output': None,
						'input_tokens': 0,
						'output_tokens': 0,
						'latency_seconds': 0,
						'status': 'failed',
						'error': str(error),
					}
				)

	pd.DataFrame(results).to_csv(output_file, index=False)

	print(f'\nResults saved to: {output_file}')
