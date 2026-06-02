# import mlflow
# import dagshub
# import json
# from mlflow import MlflowClient

# dagshub.init(repo_owner='ayushdev1905', repo_name='Food-Delivery-Time-Prediction', mlflow=True)


# # set the mlflow tracking server
# mlflow.set_tracking_uri("https://dagshub.com/ayushdev1905/Food-Delivery-Time-Prediction.mlflow")

# def load_model_information(file_path):
#     with open(file_path) as f:
#         run_info = json.load(f)
        
#     return run_info

# print("Tracking URI:", mlflow.get_tracking_uri())
# print("Model name:", model_name)

# for rm in client.search_registered_models():
#     print("Registered model:", rm.name)

# # get model name
# model_name = load_model_information("run_information.json")["model_name"]
# stage = "Staging"

# # get the latest version from staging stage
# client = MlflowClient()

# # get the latest version of model in staging
# latest_versions = client.get_latest_versions(name=model_name,stages=[stage])

# latest_model_version_staging = latest_versions[0].version

# # promotion stage
# promotion_stage = "Production"

# client.transition_model_version_stage(
#     name=model_name,
#     version=latest_model_version_staging,
#     stage=promotion_stage,
#     archive_existing_versions=True
# )


import mlflow
import dagshub
import json
from mlflow import MlflowClient

dagshub.init(
    repo_owner='ayushdev1905',
    repo_name='Food-Delivery-Time-Prediction',
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/ayushdev1905/Food-Delivery-Time-Prediction.mlflow"
)

def load_model_information(file_path):
    with open(file_path) as f:
        return json.load(f)

# Load model info first
model_name = load_model_information("run_information.json")["model_name"]
stage = "Staging"

# Create client
client = MlflowClient()

# Debug output
print("Tracking URI:", mlflow.get_tracking_uri())
print("Model name:", model_name)

print("Registered models visible from CI:")
for rm in client.search_registered_models():
    print("-", rm.name)

# Existing logic
latest_versions = client.get_latest_versions(
    name=model_name,
    stages=[stage]
)

latest_model_version_staging = latest_versions[0].version

client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version_staging,
    stage="Production",
    archive_existing_versions=True
)