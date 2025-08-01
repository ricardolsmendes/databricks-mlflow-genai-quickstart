import logging

import dotenv

from sentence_completion.evaluation import runners

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

runners.BasicPromptRunner.run()
