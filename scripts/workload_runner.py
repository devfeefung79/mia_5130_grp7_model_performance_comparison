import json
from pathlib import Path

import pandas as pd
from bedrock_client import invoke_model

ROOT = Path(__file__).resolve().parent.parent

MODELS = ['nova_lite', 'nova_micro', 'claude_sonnet', 'llama_scout']


def run_workload(workload, input_file):

	prompt_file = ROOT / 'prompts' / f'{workload}.txt'
	output_dir = ROOT / 'data' / 'outputs'
	output_file = output_dir / f'{workload}_results.csv'

	# Load input data and prompt
	df = pd.read_csv(input_file)
	prompt_template = prompt_file.read_text(encoding='utf-8')

	output_dir.mkdir(parents=True, exist_ok=True)

	results = []

	for _, row in df.iterrows():
		input_data = row.to_dict()

		prompt = prompt_template.replace(
			'{product_data}', json.dumps(input_data, ensure_ascii=False, default=str)
		)

		for model in MODELS:
			print(f'Running {workload} | {model}')

			try:
				result = invoke_model(model_name=model, prompt=prompt)

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
					}
				)

			except Exception as error:
				print(f'Error: {error}')

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
					}
				)

	pd.DataFrame(results).to_csv(output_file, index=False)

	print(f'\nResults saved to: {output_file}')
