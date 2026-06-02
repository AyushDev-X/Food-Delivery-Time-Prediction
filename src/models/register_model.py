import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging


# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# initialize dagshub
dagshub.init(repo_owner='ayushdev1905', 
             repo_name='Food-Delivery-Time-Prediction', 
             mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/ayushdev1905/Food-Delivery-Time-Prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info

def get_production_mae(client, model_name):
    try:
        versions = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        if not versions:
            return None

        version = versions[0]

        mv = client.get_model_version(
            model_name,
            version.version
        )

        return float(mv.tags["mae"])

    except Exception:
        return None


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    
    # run information file path
    run_info_path = root_path / "run_information.json"
    
    # register the model
    run_info = load_model_information(run_info_path)
    
    # get the run id
    # run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    model_registry_path = run_info["model_uri"]

    new_mae = run_info["test_mae"]

    client = MlflowClient()
    
    production_mae = get_production_mae(client,model_name)

    if production_mae is None:
        logger.info("No production model found")
        should_register = True

    elif new_mae < production_mae:
        logger.info(
            f"New MAE ({new_mae}) is better than production ({production_mae})"
        )
        should_register = True

    else:
        logger.info(
            f"Production model ({production_mae}) is better"
        )
        should_register = False

    if not should_register:
        logger.info("Skipping model registration")
        raise SystemExit(0)


    # register the model
    model_version = mlflow.register_model(model_uri=model_registry_path, name=model_name)

    # get the model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name

    client.set_model_version_tag(model_name, model_version.version, "mae", str(new_mae))
    logger.info(f"The latest model version in model registry is {registered_model_version}")
    
    # update the stage of the model to staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Staging"
    )
    
    logger.info("Model pushed to Staging stage")
    