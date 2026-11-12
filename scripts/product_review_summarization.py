from pathlib import Path

from workload_runner import run_workload

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / 'data' / 'processed' / 'product_reviews_utf8.csv'

if __name__ == '__main__':
	run_workload(workload='product_review_summarization', input_file=INPUT_FILE)
