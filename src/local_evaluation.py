import logging

import dotenv

from sentence_completion.evaluation import runners

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

result = runners.BasicPromptRunner.run()
print(result.metrics)
print(result.tables["eval_results"])
