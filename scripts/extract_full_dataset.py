import tarfile
from pathlib import Path

import pandas as pd

# =====================================
# PATH CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent

TAR_FILE = ROOT / 'data' / 'abo_dataset' / 'abo-listings.tar'

EXTRACT_DIR = ROOT / 'data' / 'abo_dataset'

OUTPUT_DIR = ROOT / 'data' / 'processed'

OUTPUT_FILE = OUTPUT_DIR / 'abo_products.csv'


# =====================================
# 1. EXTRACT TAR FILE
# =====================================


def extract_dataset():

	print('Extracting ABO dataset...')

	if not TAR_FILE.exists():
		raise FileNotFoundError(f'Cannot find archive: {TAR_FILE}')

	with tarfile.open(TAR_FILE, 'r:*') as tar:
		# Basic path safety check
		for member in tar.getmembers():
			target = (EXTRACT_DIR / member.name).resolve()

			if not target.is_relative_to(EXTRACT_DIR.resolve()):
				raise ValueError(f'Unsafe archive path: {member.name}')

		tar.extractall(EXTRACT_DIR)

	print('Extraction completed!')


# =====================================
# 2. FIND METADATA FILES
# =====================================


def find_metadata_files():

	supported = {'.csv', '.json', '.jsonl', '.gz'}

	files = []

	for file in EXTRACT_DIR.rglob('*'):
		if file.is_file():
			if file.suffix.lower() in supported:
				files.append(file)

	return files


# =====================================
# 3. LOAD METADATA
# =====================================


def load_metadata(files):

	if not files:
		raise FileNotFoundError('No supported metadata files found.')

	for file in files:
		print(f'\nTrying to load: {file.name}')

		try:
			if file.name.endswith('.jsonl.gz'):
				df = pd.read_json(file, lines=True, compression='gzip')

			elif file.suffix == '.jsonl':
				df = pd.read_json(file, lines=True)

			elif file.suffix == '.json':
				df = pd.read_json(file)

			elif file.suffix == '.csv':
				df = pd.read_csv(file)

			elif file.suffix == '.gz':
				df = pd.read_json(file, lines=True, compression='gzip')

			else:
				continue

			print('Successfully loaded!')
			return df

		except Exception as error:
			print(f'Skipped: {error}')

	raise ValueError('Could not load any metadata file.')


# =====================================
# 4. SAVE DATASET
# =====================================


def save_dataset(df):

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	df.to_csv(OUTPUT_FILE, index=False)

	print('\nDataset saved successfully!')
	print(f'Location: {OUTPUT_FILE}')


# =====================================
# 5. MAIN
# =====================================

if __name__ == '__main__':
	extract_dataset()

	metadata_files = find_metadata_files()

	print('\nMetadata files found:')

	for file in metadata_files:
		print(file.name)

	df = load_metadata(metadata_files)

	print('\nDataset Information')
	print('-------------------')

	print('Total products:', len(df))
	print('Total columns:', len(df.columns))

	print('\nAvailable columns:')
	print(df.columns.tolist())

	print('\nFirst 5 records:')
	print(df.head())

	save_dataset(df)

	print('\nPreparation completed!')
