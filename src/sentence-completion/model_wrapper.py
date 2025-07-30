import uuid
from typing import Any, Optional

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types import agent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext

import agent


class MLflowChatAgentWrapper(ChatAgent):
    """
    Wrap the custom agent in an MLflow ChatAgent,
    defining a predictable API for the agent using a signature.
    """

    def __init__(self):
        self.agent = agent.SentenceCompletionAgent()

    @mlflow.trace
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # ChatAgent has a built-in helper method to help convert framework-specific
        # messages, like mlflow ChatAgentMessage to a python dictionary.
        message_dict = self._convert_messages_to_dict(messages)
        output = self.agent.invoke(message_dict)

        return ChatAgentResponse(
            messages=[
                agent.ChatAgentMessage(
                    role=output.role, content=output.content, id=str(uuid.uuid4())
                )
            ]
        )
