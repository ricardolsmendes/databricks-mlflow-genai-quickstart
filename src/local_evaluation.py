import logging

import dotenv

from sentence_completion.model_evaluation import evaluators

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

evaluators.BasicPromptEvaluator.run()
