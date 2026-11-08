import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from settings import INPUT_DIR, OUTPUT_DIR  # noqa: E402

WORKLOADS = [
	'product_catalogue_extraction',
	'customer_issue_classification',
	'product_review_summarization',
	'customer_response_drafting',
]


def compare_workload(workload):

	input_file = INPUT_DIR / f'{workload}_evaluation.csv'

	generation_file = INPUT_DIR / f'{workload}_results.csv'

	if not input_file.exists() or not generation_file.exists():
		print(f'Skipping {workload}: required files not found.')
		return

	evaluation_df = pd.read_csv(input_file)
	generation_df = pd.read_csv(generation_file)

	# Keep successful evaluations only
	evaluation_df = evaluation_df[evaluation_df['status'] == 'success']

	generation_df = generation_df[generation_df['status'] == 'success']

	# Calculate average quality score by model
	quality_summary = (
		evaluation_df.groupby('model_name')
		.agg(
			average_quality_score=('overall_score', 'mean'),
			successful_evaluations=('overall_score', 'count'),
		)
		.reset_index()
	)

	# Calculate performance metrics by model
	performance_summary = (
		generation_df.groupby('model_name')
		.agg(
			average_latency_seconds=('latency_seconds', 'mean'),
			average_input_tokens=('input_tokens', 'mean'),
			average_output_tokens=('output_tokens', 'mean'),
		)
		.reset_index()
	)

	# Combine quality and performance metrics
	comparison = quality_summary.merge(performance_summary, on='model_name', how='outer')

	comparison.insert(0, 'workload', workload)

	# Round numerical values
	numeric_columns = comparison.select_dtypes(include='number').columns

	comparison[numeric_columns] = comparison[numeric_columns].round(2)

	# Sort by quality score, highest first
	comparison = comparison.sort_values(by='average_quality_score', ascending=False)

	# Save comparison CSV
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	output_file = OUTPUT_DIR / f'{workload}_model_comparison.csv'

	comparison.to_csv(output_file, index=False)

	print(f'Comparison saved: {output_file}')
	print(comparison.to_string(index=False))


def main():

	for workload in WORKLOADS:
		print(f'\nComparing: {workload}')
		compare_workload(workload)

	print('\nAll model comparisons completed.')


if __name__ == '__main__':
	main()
