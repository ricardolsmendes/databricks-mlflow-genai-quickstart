import argparse
import logging
import os

from sentence_completion.model_evaluation import evaluators

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Process Databricks job parameters.")
parser.add_argument(
    "--mlflow-experiment-id", type=str, help="The MLflow Experiment ID."
)

args = parser.parse_args()

os.environ["MLFLOW_EXPERIMENT_ID"] = args.mlflow_experiment_id

evaluators.BasicPromptEvaluator.run()
