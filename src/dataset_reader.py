"""File contains the reader utilities."""
import jsonlines
from pathlib import Path
from constants import FIELDS, DATASET_PATH


class DataReader(object):
    """Class DataSetReader helps loading dataset for the experiments."""

    def __init__(self, fpath, qas=None):
        """Check if the dataset is available for the experiments."""
        self.fpath = fpath
        path = Path(fpath)
        if not path.is_file() and path.exists() and not qas:
            raise FileNotFoundError

    def get_data_from_json(self, fields=None):
        """Retrun full dataset."""
        reader = jsonlines.open(self.fpath)
        data = []
        fields = FIELDS if not fields else fields
        for obj in reader:
            result = {}
            for field in fields:
                result[field] = obj[field]
            data.append(result)
        return data

    def get_masked_token_index_qas(self, tokenized_text, qas):
        """Return masked_index of the Named Entity."""
        named_entity = self._get_named_entity_qas(qas)
        try:
            return tokenized_text.index(named_entity)
        except:
            return None

    def _get_named_entity_qas(self, qas):
        entity = list(qas['ner'].keys())[0]
        return entity.lower()


if __name__ == "__main__":
    reader = DataReader(DATASET_PATH)
    data = reader._get_data_from_json(FIELDS)
    assert len(data) == 131969
