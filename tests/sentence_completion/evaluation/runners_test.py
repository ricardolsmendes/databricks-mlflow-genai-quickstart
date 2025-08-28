import unittest
from unittest import mock

from sentence_completion.evaluation import runners


class TestBasicPromptEvaluator(unittest.TestCase):
    @mock.patch("sentence_completion.evaluation.runners.mlflow.genai.evaluate")
    @mock.patch(
        "sentence_completion.evaluation.runners.scorers.EVAL_SCORERS", "dummy_scorers"
    )
    @mock.patch(
        "sentence_completion.evaluation.runners.dataset.EVAL_DATA", "dummy_data"
    )
    @mock.patch(
        "sentence_completion.evaluation.runners.chat_agent_wrapper.ChatAgentWrapper"
    )
    def test_run_returns_evaluation_result(self, mock_agent_wrapper, mock_evaluate):
        # Given
        mock_predict = mock.MagicMock()
        mock_agent_wrapper.return_value.predict = mock_predict
        mock_evaluate.return_value = "evaluation_result"

        # When
        result = runners.BasicPromptRunner.run()

        # Then
        mock_evaluate.assert_called_once_with(
            data="dummy_data",
            scorers="dummy_scorers",
            predict_fn=mock_predict,
        )
        self.assertEqual(result, "evaluation_result")
