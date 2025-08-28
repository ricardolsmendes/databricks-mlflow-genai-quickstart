import unittest
from unittest import mock

from sentence_completion.evaluation import runners


class TestBasicPromptEvaluator(unittest.TestCase):
    _MODULE_UNDER_TEST = "sentence_completion.evaluation.runners"

    @mock.patch(f"{_MODULE_UNDER_TEST}.mlflow.genai.evaluate")
    @mock.patch(f"{_MODULE_UNDER_TEST}.scorers.EvaluationScorersFactory.make_scorers")
    @mock.patch(f"{_MODULE_UNDER_TEST}.dataset.EVAL_DATA", "dummy_data")
    @mock.patch(f"{_MODULE_UNDER_TEST}.chat_agent_wrapper.ChatAgentWrapper")
    def test_run_returns_evaluation_result(
        self, mock_agent_wrapper, mock_make_scorers, mock_evaluate
    ):
        # Given
        mock_predict = mock.MagicMock()
        mock_agent_wrapper.return_value.predict = mock_predict
        mock_make_scorers.return_value = "dummy_scorers"
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
