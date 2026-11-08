import json
from pathlib import Path

import pandas as pd
from bedrock_client import invoke_model

ROOT = Path(__file__).resolve().parent.parent

# Nova Lite is the evaluation judge
JUDGE_MODEL = 'nova_lite'

WORKLOADS = [
	'product_catalogue_extraction',
	'customer_issue_classification',
	'product_review_summarization',
	'customer_response_drafting',
]

INPUT_DIR = ROOT / 'data' / 'outputs'
CRITERIA_DIR = ROOT / 'evaluation_criteria'
OUTPUT_DIR = ROOT / 'results' / 'evaluations'


def load_criteria(workload):
	criteria_file = CRITERIA_DIR / f'{workload}.json'

	with open(criteria_file, 'r', encoding='utf-8') as file:
		criteria_data = json.load(file)

	return criteria_data


def build_judge_prompt(input_data, generated_output, criteria_data):
	prompt_file = ROOT / 'prompts' / 'llm_judge.txt'

	prompt_template = prompt_file.read_text(encoding='utf-8')

	return prompt_template.format(
		workload=criteria_data['workload'],
		input_data=input_data,
		generated_output=generated_output,
		scale=json.dumps(criteria_data['scale'], ensure_ascii=False, indent=2),
		criteria=json.dumps(criteria_data['criteria'], ensure_ascii=False, indent=2),
	)


def parse_judge_output(output):
	"""
	Extract JSON from the judge response.
	"""
	output = output.strip()

	# Remove Markdown code fences if present
	if output.startswith('```'):
		output = output.replace('```json', '', 1)
		output = output.replace('```', '').strip()

	return json.loads(output)


def evaluate_workload(workload):
	input_file = INPUT_DIR / f'{workload}_results.csv'
	output_file = OUTPUT_DIR / f'{workload}_evaluation.csv'

	if not input_file.exists():
		print(f'Skipping {workload}: result file not found.')
		return

	print(f'\nEvaluating workload: {workload}')

	df = pd.read_csv(input_file)
	criteria_data = load_criteria(workload)

	results = []

	for index, row in df.iterrows():
		if row.get('status') != 'success':
			print(f'Skipping failed generation at row {index}.')
			continue

		input_data = row.get('input_data', '')
		generated_output = row.get('output', '')

		if pd.isna(input_data) or pd.isna(generated_output):
			print(f'Skipping incomplete row {index}.')
			continue

		prompt = build_judge_prompt(
			input_data=input_data, generated_output=generated_output, criteria_data=criteria_data
		)

		model_name = row.get('model_name', 'unknown')
		print(f'Judging {index + 1}/{len(df)} | {workload} | {model_name}')

		try:
			judge_response = invoke_model(
				model_name=JUDGE_MODEL, prompt=prompt, max_tokens=1500, temperature=0.0
			)

			evaluation = parse_judge_output(judge_response['output'])

			results.append(
				{
					'workload': workload,
					'model_name': model_name,
					'input_data': input_data,
					'generated_output': generated_output,
					'scores': json.dumps(evaluation.get('scores', []), ensure_ascii=False),
					'overall_score': evaluation.get('overall_score'),
					'overall_comment': evaluation.get('overall_comment'),
					'judge_model': JUDGE_MODEL,
					'judge_latency_seconds': judge_response['latency_seconds'],
					'status': 'success',
				}
			)

		except Exception as error:
			print(f'Evaluation error: {error}')

			results.append(
				{
					'workload': workload,
					'model_name': model_name,
					'input_data': input_data,
					'generated_output': generated_output,
					'scores': None,
					'overall_score': None,
					'overall_comment': str(error),
					'judge_model': JUDGE_MODEL,
					'judge_latency_seconds': None,
					'status': 'failed',
				}
			)

		# Save progress after each evaluation
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

		pd.DataFrame(results).to_csv(output_file, index=False)

	print(f'Evaluation saved to: {output_file}')


def main():
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	for workload in WORKLOADS:
		evaluate_workload(workload)

	print('\nAll workload evaluations completed.')


if __name__ == '__main__':
	main()
