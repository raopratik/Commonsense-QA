import sys

sys.path.append(".")
import pandas as pd
from transformers import DistilBertTokenizer
from tqdm import tqdm
from SocialIQA.utils import save_dictionary, load_dictionary
from torch.utils import data
from SocialIQA import constants as con
from SocialIQA.Preprocess.Dataloaders import DownstreamDataset


class Preprocessor:

    def __init__(self):
        self.train_path = "Downstream/Data/train_pandas.csv"
        self.valid_path = "Downstream/Data/dev_pandas.csv"

        self.train_data = None
        self.valid_data = None
        self.tokenizer = None

        self.train_loaders = None
        self.valid_loaders = None

        self.input_dict = {
            "train": {
                "ans": [],
                "label": []
            },
            "valid": {
                "ans": [],
                "label": []
            },
        }

    def setup_tokenizer(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',
                                                             max_length=256, padding="max_length",
                                                             truncation=True, is_split_into_words=False)

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.valid_data = pd.read_csv(self.valid_path)

    def tokenize_data(self, data, key):
        print("Tokenizing", key, "data...")
        questions = data['question']
        labels = data['labels']
        contexts = data['context']

        for i in tqdm(range(len(data))):
            question = self.tokenizer(text=questions[i])['input_ids']
            context = self.tokenizer(text=contexts[i])['input_ids'][1:]
            ans_a = self.tokenizer(text=data["ans_a"][i])['input_ids'][1:]
            ans_b = self.tokenizer(text=data["ans_b"][i])['input_ids'][1:]
            ans_c = self.tokenizer(text=data["ans_c"][i])['input_ids'][1:]

            input_line = question + context + ans_a + ans_b + ans_c
            self.input_dict[key]["ans"].append(input_line + [0 for _ in range(512 - len(input_line))])
            self.input_dict[key]["label"].append(labels[i])

    def get_loaders(self, load_from_pkl=False):
        if load_from_pkl:
            try:
                self.input_dict = load_dictionary("Downstream/Data/input_dict.pkl")
            except Exception as e:
                print(e)

        train_dataset = DownstreamDataset(input_dict=self.input_dict["train"])
        valid_dataset = DownstreamDataset(input_dict=self.input_dict["valid"])
        loader_args = dict(shuffle=True, batch_size=con.BATCH_SIZE)
        self.train_loaders = data.DataLoader(train_dataset, **loader_args)
        self.valid_loaders = data.DataLoader(valid_dataset, **loader_args)

    def setup(self):
        self.load_data()
        self.setup_tokenizer()
        self.tokenize_data(data=self.train_data, key="train")
        self.tokenize_data(data=self.valid_data, key="valid")
        save_dictionary(dictionary=self.input_dict,
                        save_path="Downstream/Data/input_dict.pkl")
        self.get_loaders(load_from_pkl=True)


if __name__ == '__main__':
    p = Preprocessor()
    p.setup()
