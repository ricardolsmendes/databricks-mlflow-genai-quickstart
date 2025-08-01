from mlflow.types import agent
from mlflow.types.agent import ChatAgentMessage


class SentenceCompletionMessageFactory:
    """Factory for creating ChatAgentMessage instances."""

    _SYSTEM_PROMPT = (
        "You are a smart bot that can complete sentence templates to make them funny."
        " Be creative and edgy."
    )

    @classmethod
    def make_input_messages(cls, sentence_template: str) -> list[ChatAgentMessage]:
        """Create the input messages for a sentence completion chat."""
        return [
            agent.ChatAgentMessage(role="system", content=cls._SYSTEM_PROMPT),
            agent.ChatAgentMessage(role="user", content=sentence_template),
        ]
