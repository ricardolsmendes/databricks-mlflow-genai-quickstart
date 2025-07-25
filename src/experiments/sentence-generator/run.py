import dotenv
from mlflow.types import agent

import prompts
import sentences_agent


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
        role="system", content=prompts.SENTENCE_GENERATION_SYSTEM_PROMPT
    ),
    agent.ChatAgentMessage(role="user", content=SENTENCE_TEMPLATE),
]

result = sentences_agent.SentencesAgent().predict(messages)
print(f"Input: {SENTENCE_TEMPLATE}")
print(f"Output: {result.messages[0].content}")
