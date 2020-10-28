import sys

sys.path.append(".")
from CommonSense.utils import load_dictionary
from transformers import BertTokenizer
from tqdm import tqdm


class Preprocessor:
    options = ['A', 'B', 'C', 'D', 'E']
    max_seq_len = 64

    def __init__(self):
        self.input_dict = load_dictionary("CommonSense/Data/input_dict.pkl")
        self.tokenized_dict = {
            "train":{
                opt: [] for opt in Preprocessor.options + ['question', 'y', 'concept']
                                   + [x + "_att" for x in Preprocessor.options]

            },
            "dev": {
                opt: [] for opt in Preprocessor.options + ['question', 'y', 'concept']
                                   + [x + "_att" for x in Preprocessor.options]
            }
        }
        self.tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_data(self, data_dict, key="train"):
        for i in tqdm(range(len(data_dict['question']))):
            question = self.tokenzier(data_dict['question'][i])
            context = self.tokenzier(data_dict['concept'][i])[1:]

            for option in Preprocessor.options:
                answer = self.tokenzier(data_dict[option][i])[1:]
                input_line = question + context + answer

                self.tokenized_dict[key][option].append(input_line + [0 for _ in range(Preprocessor.max_seq_len - len(input_line))])
                self.tokenized_dict[key][option + '_att'].append([1 for _ in range(len(input_line))] + [0 for _ in range(Preprocessor.max_seq_len - len(input_line))])
            self.input_dict[key]["label"].append(data_dict['y'][i])

    def setup(self):
        self.tokenize_data(data_dict=self.input_dict['train'], key='train')
        self.tokenize_data(data_dict=self.input_dict['dev'], key='dev')












