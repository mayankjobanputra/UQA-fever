"""This file contains code that will verify facts from the dataset."""

import jsonlines
from dataset_reader import DataReader
from tokenizer import Tokenizer
from predictor import Predictor
from constants import DATASET_PATH, MODEL_PATH, RESULT_PATH
from utils import get_bar


class FVR(object):
    """FVR (Fact Verifier) will generate the accuracy score."""

    def __init__(self, data_path, model_path):
        """Initialize the reader, tokenizer, and prediction module."""
        self.reader = DataReader(data_path)
        self.tokenizer = Tokenizer(model_path)
        self.predictor = Predictor(model_path)

    def verify_claims_and_export_raw_results(self, writer):
        """Verfies the claims from the dataset using MaskedLM and export results."""
        data = self.reader.get_data_from_json()
        bar = get_bar(len(data))
        for idx, obj in enumerate(data):
            tokenized_claim = self.tokenizer.get_tokenized_text(obj['claim'])
            for qas in obj['qas']:
                masked_index = self.reader.get_masked_token_index_qas(tokenized_claim, qas)
                if not masked_index:
                    s_tokenized_claim = self.tokenizer.get_s_tokenized_text(obj['claim'])
                    masked_index = self.reader.get_masked_token_index_qas(s_tokenized_claim, qas)
                if masked_index:
                    try:
                        qas['prediction'] = self.predictor.predict(tokenized_claim, masked_index)
                        qas['is_correct'] = qas['prediction'] == qas['a'].lower()
                    except:
                        qas['prediction'] = None
                        qas['is_correct'] = False
                    finally:
                        qas['tokenized_corectly'] = True
                else:
                    qas['prediction'] = None
                    qas['is_correct'] = False
                    qas['tokenized_corectly'] = False
            writer.write(obj)
            bar.update(idx+1)


if __name__ == "__main__":
    writer = jsonlines.open(RESULT_PATH, mode='w')
    fvr = FVR(DATASET_PATH, MODEL_PATH)
    fvr.verify_claims_and_export_raw_results(writer)
    writer.close()
