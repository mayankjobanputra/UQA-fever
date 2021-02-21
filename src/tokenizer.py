"""Tokenizer contains the methods that tokenize the claim."""

from pytorch_pretrained_bert import BertTokenizer
import spacy


class Tokenizer(object):
    """Tokenizer can use different tokenizers to tokenize the sentence."""

    def __init__(self, model_path):
        """Initialize the tokenizers."""
        self.b_tokenizer = BertTokenizer.from_pretrained(model_path)
        self.s_tokenizer = spacy.load('xx')

    def get_tokenized_text(self, text):
        """Retrun tokenized text."""
        text = text.replace(',', '[SEP]')
        text = text.replace('.', ' [SEP]')
        text = "[CLS] " + text
        return self.b_tokenizer.tokenize(text)

    def get_s_tokenized_text(self, text):
        """Retrun tokenized text."""
        tokens = self.s_tokenizer(text)
        tokens_list = ['[SEP]' if token.text in [',', '.'] else token.text.lower() for token in tokens]
        return ['[CLS]'] + tokens_list


if __name__ == "__main__":
    from constants import SAMPLE_DATA
    tokenizer = Tokenizer('/data/FEVER/')
    tokenized_text = tokenizer.get_tokenized_text(SAMPLE_DATA['claim'])
    s_tokenized_text = tokenizer.get_s_tokenized_text(SAMPLE_DATA['claim'])
    assert tokenized_text == ['[CLS]', 'keith', 'urban', 'has', 'put', 'out', 'no', 'fewer', 'than', 'nine', 'studio', 'records', '[SEP]']
    assert s_tokenized_text == tokenized_text
