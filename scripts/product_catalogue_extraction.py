from pathlib import Path

from workload_runner import run_workload

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / 'data' / 'processed' / 'selected_products.csv'

if __name__ == '__main__':
	run_workload(workload='product_catalogue_extraction', input_file=INPUT_FILE)
