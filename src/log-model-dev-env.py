import logging
import os
from datetime import datetime

import dotenv
import mlflow
from mlflow.models import resources

from sentence_completion import chat_agent_wrapper, llm_client_wrapper

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
endpoint_name = llm_client_wrapper.DatabricksEndpointClient.get_endpoint_name()

# Specify Databricks resources that the agent needs to access.
# This step lets Databricks automatically configure authentication
# so the agent can access these resources when it's deployed.
resource_list = [resources.DatabricksServingEndpoint(endpoint_name=endpoint_name)]

mlflow.models.set_model(chat_agent_wrapper.ChatAgentWrapper())
logged_agent_info = mlflow.pyfunc.log_model(
    name=f"logged_sentence_completion_{run_datetime}".replace(".", "_"),
    python_model=os.path.join(
        os.getcwd(), "src/sentence_completion/chat_agent_wrapper.py"
    ),
    registered_model_name=f""
    f"{os.environ['MODEL_CATALOG']}"
    f".{os.environ['MODEL_SCHEMA']}"
    f".{os.environ['MODEL_NAME']}",
    resources=resource_list,
)

print(logged_agent_info)
