"""Predictor contains the methods that calculate the fianl output."""

import torch
from pytorch_pretrained_bert import BertForMaskedLM
from dataset_reader import DataReader
from tokenizer import Tokenizer
from constants import MODEL_PATH


class Predictor:
    """Predictor uses BertModel to predict the Masked words of the sentence."""

    def __init__(self, name):
        """Initialize the model and tokenizer."""
        self.tokenizer = Tokenizer(MODEL_PATH)
        self.model = BertForMaskedLM.from_pretrained(name, cache_dir=None)
        self.model.eval()
        self.model.to('cuda')

    def predict(self, tokenized_text, masked_index):
        """Predict the masked word."""
        indexed_tokens = self.tokenizer.b_tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = []
        for token in tokenized_text:
            seg_id = 0
            segments_ids.append(seg_id)
            seg_id += 1 if token == "[SEP]" else 0

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')

        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = self.tokenizer.b_tokenizer.convert_ids_to_tokens([predicted_index])[0]
        return predicted_token


if __name__ == "__main__":
    from constants import SAMPLE_DATA
    reader = DataReader("", SAMPLE_DATA)
    predictor = Predictor(MODEL_PATH)
    tokenizer = Tokenizer(MODEL_PATH)
    tokenized_text = tokenizer.get_tokenized_text(SAMPLE_DATA['claim'])
    masked_index = reader.get_masked_token_index_qas(tokenized_text, SAMPLE_DATA['qas'][2])
    predicted_token = predictor.predict(tokenized_text, masked_index)
    assert predicted_token == SAMPLE_DATA['qas'][2]['a']
