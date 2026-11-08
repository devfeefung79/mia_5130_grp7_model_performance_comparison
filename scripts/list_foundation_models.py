import boto3
from dotenv import load_dotenv

load_dotenv()

bedrock_client = boto3.client('bedrock', region_name='us-east-2')

models = bedrock_client.list_foundation_models()

SELECTED_MODEL_LIST = [
	'Claude Sonnet 4.6',
	'Qwen3 32B (dense)',
	'Llama 4 Scout 17B Instruct',
	'Nova Lite',
	'Nova Micro',
]

for model in models['modelSummaries']:
	if model['modelName'] in SELECTED_MODEL_LIST:
		print(f'modelId: {model["modelId"]}, modelName: {model["modelName"]}')
