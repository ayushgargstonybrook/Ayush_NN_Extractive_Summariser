from allennlp.common import JsonDict
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('summary-classifier')
class SummaryPredictor(Predictor):
    """Predictor wrapper for the AcademicPaperClassifier"""

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        content = json_dict['content']

        instance = self._dataset_reader.text_to_instance(content=content)
        # print(instance)
        # print(self._model.forward_on_instance(instance))
        # print("Instance: ", instance)
        relevant_dict = self._model.vocab.get_index_to_token_vocabulary('relevant')
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [relevant_dict[i] for i in range(len(relevant_dict))]

        return {"instance": self.predict_instance(instance), "all_labels": all_labels}
