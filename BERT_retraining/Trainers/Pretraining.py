import sys

sys.path.append(".")

import torch
from BERT_retraining.DataSetup.Preprocessors import Preprocessors
from BERT_retraining.Models.DistilBERT import PretrainingModel
from BERT_retraining import constants as con
from torch import optim
from tqdm import tqdm


class PretrainingTrainer:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.optimizer = None

    def setup_preprocessed_data(self):
        self.preprocessor = Preprocessors()
        self.preprocessor.get_loaders(load_flag=True)

    def setup_model(self):
        # Create multilingual vocabulary
        self.model = PretrainingModel()

        if con.CUDA:
            self.model = self.model.cuda()

    def setup_scheduler_optimizer(self):
        lr_rate = 2e-5
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr_rate, weight_decay=0)

    def train_model(self):
        train_loader = self.preprocessor.train_loaders

        self.model.train()
        total_loss = 0
        batch_correct = 0
        total_correct = 0
        index = 0
        for input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
            next_sentence_labels in train_loader:
            self.optimizer.zero_grad()
            if con.CUDA:
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()
                masked_lm_positions = masked_lm_positions.cuda()
                masked_lm_ids = masked_lm_ids.cuda()
                masked_lm_weights = masked_lm_weights.cuda()
                next_sentence_labels = next_sentence_labels.cuda()

            masked_outputs, next_sentence_outputs, mlm_loss, nsp_loss = self.model(input_ids, masked_lm_ids,
                                                                                   masked_lm_positions,
                                                                                   next_sentence_labels)
            batch_loss = mlm_loss + nsp_loss
            total_loss += batch_loss

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            batch_correct += self.evaluate(masked_outputs=masked_outputs,
                                           masked_lm_ids=masked_lm_ids)
            total_correct += (con.BATCH_SIZE * 5)
            index += 1

            if index % 200 == 0:
                print("Running train loss", batch_loss)
                print("Running train acc", batch_correct / total_correct)

        print("Total train loss:", total_loss / index)
        print("Total train acc:", batch_correct / total_correct)
        print('------------------------')
        self.validation()

    def validation(self):
        valid_loader = self.preprocessor.valid_loaders
        self.model.eval()
        total_loss = 0
        batch_correct = 0
        total_correct = 0
        index = 0

        for input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
            next_sentence_labels in valid_loader:
            self.optimizer.zero_grad()
            if con.CUDA:
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()
                masked_lm_positions = masked_lm_positions.cuda()
                masked_lm_ids = masked_lm_ids.cuda()
                masked_lm_weights = masked_lm_weights.cuda()
                next_sentence_labels = next_sentence_labels.cuda()

            masked_outputs, next_sentence_outputs, mlm_loss, nsp_loss = self.model(input_ids, masked_lm_ids,
                                                                                   masked_lm_positions,
                                                                                   next_sentence_labels)
            batch_loss = mlm_loss + nsp_loss
                total_loss += batch_loss.cpu().detach().numpy()
            batch_correct += self.evaluate(masked_outputs=masked_outputs, masked_lm_ids=masked_lm_ids)
            total_correct += (con.BATCH_SIZE * 5)
            index += 1

        print("Total valid loss:", total_loss / index)
        print("Total valid acc:", batch_correct / total_correct)
        print('------------------------')

    def evaluate(self, masked_outputs, masked_lm_ids):
        masked_output_predictions = torch.argmax(masked_outputs, dim=-1).view(-1)
        masked_lm_ids = masked_lm_ids.view(-1)
        correct = torch.sum(torch.eq(masked_output_predictions, masked_lm_ids))
        return correct.cpu().detach().numpy()

    def run_pretraining(self):
        self.setup_preprocessed_data()
        self.setup_model()
        self.setup_scheduler_optimizer()
        for epoch in range(30):
            self.train_model()
            torch.save(self.model.distil.state_dict(), "BERT_retraining/Data/core_model" + str(epoch))
            torch.save(self.model.state_dict(), "BERT_retraining/Data/mlm_nsp_model" + str(epoch))


if __name__ == '__main__':
    trainer = PretrainingTrainer()
    trainer.run_pretraining()

