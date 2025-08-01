import unittest

from mlflow.types import agent

import sentence_completion.message_factory as message_factory


class TestSentenceCompletionMessageFactory(unittest.TestCase):

    def test_make_input_messages_returns_system_and_user_messages(self):
        # Given
        template = "This is a test template"

        # When
        messages = message_factory.SentenceCompletionMessageFactory.make_input_messages(
            template
        )

        # Then
        self.assertEqual(len(messages), 2)

        system_message = messages[0]
        self.assertIsInstance(system_message, agent.ChatAgentMessage)
        self.assertEqual(system_message.role, "system")

        user_message = messages[1]
        self.assertIsInstance(user_message, agent.ChatAgentMessage)
        self.assertEqual(user_message.role, "user")
        self.assertEqual(user_message.content, template)
