from typing import Iterable

import mlflow
import openai
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class SentenceCompletionAgent:

    def __init__(self):
        # Enable MLflow's autologging to instrument the application with Tracing.
        mlflow.openai.autolog()

        # Connect to a Databricks LLM via OpenAI using the same credentials as MLflow.
        # Alternatively, you can use your own OpenAI credentials here.
        mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
        self.client = openai.OpenAI(
            api_key=mlflow_creds.token,
            base_url=f"{mlflow_creds.host}/serving-endpoints",
        )

    def invoke(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        """Complete a sentence template using an LLM."""

        response = self.client.chat.completions.create(
            # This example uses Databricks hosted Llama 4 Maverick. If you provide your
            # own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o.
            model="databricks-llama-4-maverick",
            messages=messages,
        )
        return response.choices[0].message
