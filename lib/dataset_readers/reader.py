import json

from allennlp.data import TokenIndexer, DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, ListField, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from typing import Iterator, List, Dict
import numpy as np


@DatasetReader.register("summariser_dataset_reader")
class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, content: List[str], relevant: List[int]=None) -> Instance:
        content_list = []
        for sentence in content:
            temp = []
            for word in sentence.split():
                temp.append(Token(word))
            content_list.append(temp)

        content_list = [TextField(sentence, self.token_indexers) for sentence in content_list]
        content_field = ListField(content_list)
        relevant_field = ArrayField(np.array(relevant))

        fields = {"content": content_field, "relevant": relevant_field}

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                json_line = json.loads(line)
                content = json_line['content']
                relevant = json_line['relevant']
                yield self.text_to_instance(content, relevant)
