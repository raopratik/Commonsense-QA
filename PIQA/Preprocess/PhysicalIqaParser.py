import sys
sys.path.append(".")
import pandas as pd
import json


class PhysicalIqaParser:

    def __init__(self):
        # File paths
        self.train_json_path = "Data/train.jsonl"
        self.dev_json_path = "Data/valid.jsonl"
        self.train_labels_path = "Data/train-labels.lst"
        self.dev_labels_path = "Data/valid-labels.lst"

    def read_json(self, file_path):
        with open(file_path, 'r') as fh:
            result = [json.loads(x) for x in fh.readlines()]
        return result

    def read_labels(self, file_path):
        with open(file_path, 'r') as fh:
            result = [int(x) for x in fh.readlines()]
        return result

    def convert_json_to_pandas(self, json_data):

        json_dict = {"sol1": [],
                     "sol2": [],
                     "goal": []
                     }
        for json_line in json_data:
            json_dict["sol1"].append(json_line['sol1'])
            json_dict["sol2"].append(json_line['sol2'])
            json_dict["goal"].append(json_line['goal'])

        return pd.DataFrame.from_dict(json_dict)

    def merge_x_and_labels(self, pandas_x, labels):
        pandas_x['labels'] = labels
        return pandas_x

    def setup(self):
        json_data_train = self.read_json(self.train_json_path)
        json_data_dev = self.read_json(self.dev_json_path)
        train_y = self.read_labels(self.train_labels_path)
        dev_y = self.read_labels(self.dev_labels_path)
        train_pandas_x = self.convert_json_to_pandas(json_data_train)
        dev_pandas_x = self.convert_json_to_pandas(json_data_dev)
        train_pandas = self.merge_x_and_labels(train_pandas_x, train_y)
        dev_pandas = self.merge_x_and_labels(dev_pandas_x, dev_y)
        pd.DataFrame.to_csv(train_pandas, 'Data/train_pandas.csv')
        pd.DataFrame.to_csv(dev_pandas, 'Data/dev_pandas.csv')


if __name__ == '__main__':
    s = PhysicalIqaParser()
    s.setup()

