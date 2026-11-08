from pathlib import Path

from workload_runner import run_workload

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / 'data' / 'processed' / 'customer_issues.csv'

if __name__ == '__main__':
	run_workload(workload='customer_issue_classification', input_file=INPUT_FILE)
