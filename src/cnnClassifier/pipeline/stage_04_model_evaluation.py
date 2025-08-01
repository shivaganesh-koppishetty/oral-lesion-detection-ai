from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_with_mlflow import Evaluation
from cnnClassifier import logger


STAGE_NAME = "Model Evaluation"

class EvaluationPipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(evaluation_config)
        evaluation._valid_generator()
        evaluation.evaluation()
        evaluation.log_into_mlflow()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e 
