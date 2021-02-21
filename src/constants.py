SAMPLE_DATA = {"id": 90874, "verifiable": "VERIFIABLE", "label": "SUPPORTS", "claim": "Keith Urban has put out no fewer than nine studio records.", "qas": [{"q": "<BLANK> Urban has put out no fewer than nine studio records.", "a": "Keith", "ner": {"Keith": "PERSON"}}, {"q": "Keith <BLANK> has put out no fewer than nine studio records.", "a": "Urban", "ner": {"Urban": "PERSON"}}, {"q": "Keith Urban has put out no fewer than <BLANK> studio records.", "a": "nine", "ner": {"nine": "NUMBER"}}]}
DATASET_PATH = '/data/FEVER/output_fever_dev.jsonl'
MODEL_PATH = '/data/MODELS/BERT_LARGE_UC/'
FIELDS = ["id", "verifiable", "label", "claim", "qas"]
RESULT_PATH = '/data/FEVER/results/dev_large_bert_token_better_gpu_uncased_predictions.jsonl'
THRESHOLD = 0.5

BEST_RESULT = '/data/FEVER/results/large_bert_token_better_gpu_uncased_predictions.jsonl'
