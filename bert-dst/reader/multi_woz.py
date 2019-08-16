import json
from pytorch_transformers.tokenization_bert import BertTokenizer


class MultiWOZReader:
    def __init__(self, vocab_file):
        self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file)

    def read(self, input_file):
        pass

