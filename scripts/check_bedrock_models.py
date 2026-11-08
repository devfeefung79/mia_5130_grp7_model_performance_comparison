import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

REGION = 'us-east-2'

# SELECTED_MODEL_LIST = [
#     "Claude Sonnet 4.6",
#     "Llama 4 Scout 17B Instruct",
#     "Nova Lite",
#     "Nova Micro",
# ]

bedrock_client = boto3.client('bedrock', region_name=REGION)


def main():
	response = bedrock_client.list_foundation_models()
	models = response.get('modelSummaries', [])

	# # 2. Filter selected models
	# selected_models = [model for model in models if model['modelName'] in SELECTED_MODEL_LIST]

	print(f'\nRegion: {REGION}')
	print(f'Selected models found: {len(models)}\n')

	# 3. Check model access
	for model in models:
		model_id = model['modelId']
		model_name = model['modelName']

		# print('=' * 80)
		# print(f'Model: {model_name}')
		# print(f'Model ID: {model_id}')

		try:
			availability = bedrock_client.get_foundation_model_availability(modelId=model_id)

			agreement = availability.get('agreementAvailability', {})

			# print(f'Agreement: {agreement.get("status")}')
			# print(f'Authorization: {availability.get("authorizationStatus")}')
			# print(f'Entitlement: {availability.get("entitlementAvailability")}')
			# print(f'Region: {availability.get("regionAvailability")}')

			# Overall access indication
			accessible = (
				agreement.get('status') == 'AVAILABLE'
				and availability.get('authorizationStatus') == 'AUTHORIZED'
				and availability.get('entitlementAvailability') == 'AVAILABLE'
				and availability.get('regionAvailability') == 'AVAILABLE'
			)

			# print only available models
			if accessible:
				print(f'Model: {model_name}, Model ID: {model_id}, ACCESS STATUS: AVAILABLE')

			# print(f'ACCESS STATUS: {"AVAILABLE" if accessible else "NOT AVAILABLE"}')

		except ClientError as e:
			print(f'ACCESS CHECK ERROR: {e}')


if __name__ == '__main__':
	main()
