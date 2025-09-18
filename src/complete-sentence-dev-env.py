import logging

import dotenv

from sentence_completion import chat_agent_wrapper, message_factory

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

SENTENCE_TEMPLATE = (
    "Yesterday, ____ (person) brought a ____ (item) and used it to ____ (verb) a"
    " ____ (object)"
)

# Complete sentences using LLMs
result = chat_agent_wrapper.ChatAgentWrapper().predict(
    message_factory.SentenceCompletionChatMessageFactory.make_conversation_starters(
        SENTENCE_TEMPLATE
    )
)
logging.info(f"Input: {SENTENCE_TEMPLATE}")
logging.info(f"Output: {result.messages[0].content}")
