from mlflow import deployments


class DatabricksEndpointClient:
    """
    Wrap a Databricks Serving Endpoint client,
    defining a predictable API for LLM models served by Databricks.
    """

    def __init__(self, name: str = "databricks-llama-4-maverick"):
        self._client = deployments.get_deploy_client("databricks")
        self._endpoint_name = name

    def predict(self, messages: list[dict]) -> dict:
        response = self._client.predict(
            endpoint=self._endpoint_name, inputs={"messages": messages}
        )

        return response["choices"][0]["message"]
