import mlflow

from sentence_completion import agent_wrapper
from sentence_completion.model_evaluation import dataset, scorers


class BasicPromptEvaluator:
    """Evaluator for the sentence completion agent using a basic prompt."""

    @classmethod
    def run(cls) -> None:
        # Run model_evaluation
        print("Evaluating with basic prompt...")
        mlflow.genai.evaluate(
            data=dataset.EVAL_DATA,
            scorers=scorers.EVAL_SCORERS,
            predict_fn=agent_wrapper.MLflowChatAgentWrapper().predict,
        )
