import logging

import mlflow
from mlflow.models.evaluation.base import EvaluationResult

from sentence_completion import agent_wrapper
from sentence_completion.evaluation import dataset, scorers


class BasicPromptRunner:
    """Evaluate the sentence completion agent using basic prompt."""

    @classmethod
    def run(cls) -> EvaluationResult:
        logging.info("Evaluating with basic prompt...")
        return mlflow.genai.evaluate(
            data=dataset.EVAL_DATA,
            scorers=scorers.EVAL_SCORERS,
            predict_fn=agent_wrapper.MLflowChatAgentWrapper().predict,
        )
