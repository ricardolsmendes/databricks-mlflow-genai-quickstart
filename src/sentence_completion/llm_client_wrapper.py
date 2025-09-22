from mlflow import deployments


class DatabricksEndpointClient:
    """
    Wrap a Databricks Serving Endpoint client,
    defining a predictable API for LLM models served by Databricks.
    """

    _ENDPOINT_NAME = "databricks-llama-4-maverick"

    def __init__(self):
        self._client = deployments.get_deploy_client("databricks")

    @classmethod
    def get_endpoint_name(cls) -> str:
        return cls._ENDPOINT_NAME

    def predict(self, messages: list[dict]) -> dict:
        response = self._client.predict(
            endpoint=self._ENDPOINT_NAME, inputs={"messages": messages}
        )

        return response["choices"][0]["message"]
