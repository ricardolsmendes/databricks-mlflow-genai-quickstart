from mlflow import deployments


class SentenceCompletionAgent:
    _MODEL = "databricks-llama-4-maverick"

    def __init__(self):
        self._client = deployments.get_deploy_client("databricks")

    def invoke(self, messages: list[dict]) -> dict:
        """Complete a sentence template using an LLM."""

        response = self._client.predict(
            endpoint=self._MODEL, inputs={"messages": messages}
        )

        return response["choices"][0]["message"]
