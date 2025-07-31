import dotenv

import agent_wrapper
import message_factory

SENTENCE_TEMPLATE = """
Yesterday, ____ (person) brought a ____ (item) and used it to ____ (verb) a ____ (object)
"""

dotenv.load_dotenv()

input_messages = message_factory.SentenceCompletionMessageFactory.make_input_messages(
    SENTENCE_TEMPLATE
)

result = agent_wrapper.MLflowChatAgentWrapper().predict(input_messages)
print(f"Input: {SENTENCE_TEMPLATE}")
print(f"Output: {result.messages[0].content}")
