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