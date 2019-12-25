import torch
from typing import Dict
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy
from torch.nn import MSELoss
from datetime import datetime


@Model.register("summarizer-model")
class Summarizer(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 attention: Seq2SeqEncoder,
                 classifier: FeedForward) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.attention = attention

        self.classifier = classifier
        self.accuracy = BooleanAccuracy()
        self.loss = MSELoss()

    def forward(self,
                content: Dict[str, torch.Tensor],
                relevant: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        for key in content.keys():
            content[key] = torch.squeeze(content[key])

        mask = get_text_field_mask(content)
        embeddings = self.word_embeddings(content)
        encoder_out = self.encoder(embeddings, mask).unsqueeze(0)
        sentence_mask = torch.ones(1, encoder_out.shape[1])
        attentive_sentence_embeddings = self.attention(encoder_out, sentence_mask).squeeze()
        logits = self.classifier(attentive_sentence_embeddings)
        output = {"logits": logits}
        # torch.save(logits, f'{datetime.now()}.txt')
        if relevant is not None:
            self.accuracy(logits.round(), relevant.reshape(-1,1))
            output["loss"] = self.loss(logits, relevant.reshape(-1, 1))

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
