import logging

import mlflow
from mlflow.models.evaluation.base import EvaluationResult

from sentence_completion import agent_wrapper
from sentence_completion.model_evaluation import dataset, scorers


class BasicPromptEvaluator:
    """Evaluator for the sentence completion agent using basic prompts."""

    @classmethod
    def run(cls) -> EvaluationResult:
        logging.info("Evaluating with basic prompts...")
        return mlflow.genai.evaluate(
            data=dataset.EVAL_DATA,
            scorers=scorers.EVAL_SCORERS,
            predict_fn=agent_wrapper.MLflowChatAgentWrapper().predict,
        )
