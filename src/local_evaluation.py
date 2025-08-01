import dotenv

from sentence_completion.model_evaluation import evaluators

dotenv.load_dotenv()

evaluators.BasicPromptEvaluator.run()
