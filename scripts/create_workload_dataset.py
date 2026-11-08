from pathlib import Path

import pandas as pd

# =====================================
# CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / 'data' / 'processed' / 'abo_products.csv'
OUTPUT_DIR = ROOT / 'data' / 'processed'

SAMPLE_SIZE = 25
RANDOM_SEED = 42

# Candidate ABO column names
ID_COLUMNS = ['item_id', 'asin']
TITLE_COLUMNS = ['item_name', 'title']
DESCRIPTION_COLUMNS = ['bullet_point', 'description']
CATEGORY_COLUMNS = ['product_type', 'category']


# =====================================
# HELPER FUNCTIONS
# =====================================


def find_column(df, candidates):
	for column in candidates:
		if column in df.columns:
			return column
	return None


# =====================================
# LOAD DATA
# =====================================


def load_dataset():

	if not INPUT_FILE.exists():
		raise FileNotFoundError(INPUT_FILE)

	df = pd.read_csv(INPUT_FILE)

	print(f'Loaded {len(df)} products')
	print('Available columns:')
	print(df.columns.tolist())

	return df


# =====================================
# SELECT PRODUCTS
# =====================================


def select_products(df):

	id_col = find_column(df, ID_COLUMNS)
	title_col = find_column(df, TITLE_COLUMNS)
	description_col = find_column(df, DESCRIPTION_COLUMNS)

	if not id_col:
		raise ValueError('Cannot find product ID column. Please check your ABO column names.')

	df = df.dropna(subset=[id_col])
	df = df.drop_duplicates(subset=[id_col]).copy()

	# Prefer products with more complete metadata
	columns_to_check = [
		col
		for col in [
			title_col,
			description_col,
			find_column(df, ['brand']),
			find_column(df, ['color']),
			find_column(df, CATEGORY_COLUMNS),
			find_column(df, ['material']),
		]
		if col is not None
	]

	df['completeness_score'] = df[columns_to_check].notna().sum(axis=1)

	# Select products with richer metadata
	df = df.sort_values(by='completeness_score', ascending=False)

	# Random selection from the richer records
	top_candidates = df.head(min(len(df), SAMPLE_SIZE * 5))

	selected = top_candidates.sample(
		n=min(SAMPLE_SIZE, len(top_candidates)), random_state=RANDOM_SEED
	)

	selected = selected.drop(columns=['completeness_score'])

	return selected


# =====================================
# CREATE WORKLOAD DATASETS
# =====================================


def create_workload_datasets(products):

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	# 1. Product Catalogue Extraction
	products.to_csv(OUTPUT_DIR / 'product_catalogue_extraction.csv', index=False)

	# 2. Customer Issue Classification
	issue_data = products.copy()
	issue_data['customer_issue'] = ''
	issue_data['expected_category'] = ''

	issue_data.to_csv(OUTPUT_DIR / 'customer_issue_classification.csv', index=False)

	# 3. Product Review Summarization
	review_data = products.copy()
	review_data['synthetic_reviews'] = ''

	review_data.to_csv(OUTPUT_DIR / 'product_review_summarization.csv', index=False)

	# 4. Customer Response Drafting
	response_data = products.copy()
	response_data['synthetic_customer_enquiry'] = ''
	response_data['expected_response_points'] = ''

	response_data.to_csv(OUTPUT_DIR / 'customer_response_drafting.csv', index=False)

	# Save common selected products
	products.to_csv(OUTPUT_DIR / 'selected_products.csv', index=False)

	print('\nAll workload datasets created successfully!')


# =====================================
# MAIN
# =====================================

if __name__ == '__main__':
	dataset = load_dataset()

	selected_products = select_products(dataset)

	create_workload_datasets(selected_products)

	print(f'\nSelected {len(selected_products)} products')
