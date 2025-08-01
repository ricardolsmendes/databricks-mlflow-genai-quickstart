import dotenv

from sentence_completion import agent_wrapper, message_factory

dotenv.load_dotenv()

SENTENCE_TEMPLATE = (
    "Yesterday, ____ (person) brought a ____ (item) and used it to ____ (verb) a"
    " ____ (object)"
)

result = agent_wrapper.MLflowChatAgentWrapper().predict(
    message_factory.SentenceCompletionMessageFactory.make_input_messages(
        SENTENCE_TEMPLATE
    )
)
print(f"Input: {SENTENCE_TEMPLATE}")
print(f"Output: {result.messages[0].content}")
