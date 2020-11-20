import sys

sys.path.append("..")

import pandas as pd
from transformers import DistilBertTokenizer, BertTokenizer
from tqdm import tqdm
from PIQA.utils import save_dictionary, load_dictionary
from torch.utils import data
from PIQA import constants as con
from PIQA.Preprocess.Dataloaders_v2 import DownstreamDataset


class Preprocessor:

    def __init__(self):
        self.train_path = "Data/train_pandas.csv"
        self.valid_path = "Data/valid_pandas.csv"

        self.train_data = None
        self.valid_data = None
        self.tokenizer = None

        self.train_loaders = None
        self.valid_loaders = None

        self.input_dict = {
            "train": {
                "sol1": [],
                "sol2": [],
                "sol1_att": [],
                "sol2_att": [],
                "sol1_token": [],
                "sol2_token": [],
                "goal": [],
                "label": []
            },
            "valid": {
                "sol1": [],
                "sol2": [],
                "sol1_token": [],
                "sol2_token": [],
                "sol1_att": [],
                "sol2_att": [],
                "goal": [],
                "label": []
            },
        }

    def setup_tokenizer(self):
        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',
        #                                                      max_length=256, padding="max_length",
        #                                                      truncation=True, is_split_into_words=False)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       max_length=256, padding="max_length",
                                                       truncation=True, is_split_into_words=False)

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.valid_data = pd.read_csv(self.valid_path)

    def tokenize_data(self, data, key):
        print("Tokenizing ", key, " data...")
        questions = data['goal']
        labels = data['labels']

        for i in tqdm(range(len(data))):
            print(questions[i])
            question = self.tokenizer(text=questions[i])['input_ids']
            for ans in ["sol1", "sol2"]:
                answer = self.tokenizer(text=data[ans][i])['input_ids'][1:]
                input_line = question + answer
                q_len = len(question)
                ans_len = len(answer)
                total = q_len + ans_len

                self.input_dict[key][ans].append(input_line + [0 for _ in range(256 - len(input_line))])
                self.input_dict[key][ans + '_att'].append(
                    [1 for _ in range(len(input_line))] + [0 for _ in range(256 - len(input_line))])

                self.input_dict[key][ans + '_token'].append([0 for _ in range(q_len)] +
                                                            [1 for _ in range(ans_len)] + [0 for _ in
                                                                                           range(256 - total)])

            self.input_dict[key]["label"].append(labels[i])

    def get_loaders(self, load_from_pkl=False):
        if load_from_pkl:
            try:
                self.input_dict = load_dictionary("Data/input_dict.pkl")
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
                        save_path="Data/input_dict.pkl")
        self.get_loaders(load_from_pkl=True)


if __name__ == '__main__':
    print("came into preprocessor")
    p = Preprocessor()
    p.setup()
