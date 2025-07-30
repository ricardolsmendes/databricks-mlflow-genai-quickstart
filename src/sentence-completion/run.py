import dotenv
import prompts
from mlflow.types import agent

import model_wrapper


SENTENCE_TEMPLATE = """
Yesterday,
 ____ (person) brought a
 ____ (item) and used it to
 ____ (verb) a
 ____ (object)
"""

dotenv.load_dotenv()

messages = [
    agent.ChatAgentMessage(
        role="system", content=prompts.SENTENCE_COMPLETION_SYSTEM_PROMPT
    ),
    agent.ChatAgentMessage(role="user", content=SENTENCE_TEMPLATE),
]

result = model_wrapper.MLflowChatAgentWrapper().predict(messages)
print(f"Input: {SENTENCE_TEMPLATE}")
print(f"Output: {result.messages[0].content}")
