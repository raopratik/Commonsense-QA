import sys

sys.path.append("..")
import torch
from PIQA.Preprocess.Preprocess_v2 import Preprocessor
from PIQA.Models.DistilBERT_v2 import DownstreamModel
from PIQA import constants as con
from torch import optim


class DownstreamTrainer:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.optimizer = None

    def setup_preprocessed_data(self):
        self.preprocessor = Preprocessor()
        self.preprocessor.get_loaders(load_from_pkl=True)

    def setup_model(self):
        # Create multilingual vocabulary
        self.model = DownstreamModel()

        if con.CUDA:
            self.model = self.model.cuda()

    def setup_scheduler_optimizer(self):
        lr_rate = 1e-5
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr_rate, weight_decay=0)

    def train_model(self):
        train_loader = self.preprocessor.train_loaders
        valid_loader = self.preprocessor.valid_loaders

        self.model.train()
        total_loss = 0
        batch_correct = 0
        total_correct = 0
        index = 0
        for sol1, sol2, sol1_att, sol2_att, \
            sol1_token, sol2_token, label in valid_loader:
            self.optimizer.zero_grad()
            if con.CUDA:
                sol1 = sol1.cuda()
                sol2 = sol2.cuda()

                sol1_att = sol1_att.cuda()
                sol2_att = sol2_att.cuda()

                sol1_token = sol1_token.cuda()
                sol2_token = sol2_token.cuda()

                label = label.cuda()

            answer_preds, qa_loss = self.model(sol1, sol2, sol1_att, sol2_att, sol1_token,
                                               sol2_token, label)
            total_loss += qa_loss.cpu().detach().numpy()
            qa_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            batch_correct += self.evaluate(answer_preds=answer_preds,
                                           labels=label)
            total_correct += con.BATCH_SIZE
            index += 1

            # if index % 100 == 0:
            #     print("Running train loss", total_loss / index)
            #     print("Running train acc", batch_correct / total_correct)

        print("Train loss:", total_loss / index)
        print("Train acc:", batch_correct / total_correct)

    def valid_model(self):
        valid_loader = self.preprocessor.valid_loaders

        self.model.eval()
        total_loss = 0
        batch_correct = 0
        total_correct = 0
        index = 0
        for sol1, sol2, sol1_att, sol2_att, \
            sol1_token, sol2_token, label in valid_loader:
            if con.CUDA:
                sol1 = sol1.cuda()
                sol2 = sol2.cuda()

                sol1_att = sol1_att.cuda()
                sol2_att = sol2_att.cuda()

                sol1_token = sol1_token.cuda()
                sol2_token = sol2_token.cuda()

                label = label.cuda()

            answer_preds, qa_loss = self.model(sol1, sol2, sol1_att, sol2_att, sol1_token,
                                               sol2_token, label)
            total_loss += qa_loss.cpu().detach().numpy()
            batch_correct += self.evaluate(answer_preds=answer_preds,
                                           labels=label)
            total_correct += con.BATCH_SIZE
            index += 1

            if index % 100 == 0:
                print("Running train loss", total_loss / index)
                print("Running train acc", batch_correct / total_correct)

        print("Valid loss:", total_loss / index)
        print("Valid acc:", batch_correct / total_correct)
        print('------------------------')

    def evaluate(self, answer_preds, labels):
        labels = labels.view(-1)
        correct = torch.sum(torch.eq(answer_preds, labels))
        return correct.cpu().detach().numpy()

    def run_finetuning(self):
        self.setup_preprocessed_data()
        self.setup_model()
        self.setup_scheduler_optimizer()
        for epoch in range(10):
            print("Epoch:", epoch)
            self.train_model()
            self.valid_model()
            # torch.save(self.model.distil.state_dict(), "BERT_retraining/Data/core_model" + str(epoch))
            # torch.save(self.model.state_dict(), "BERT_retraining/Data/mlm_nsp_model" + str(epoch))


if __name__ == '__main__':
    trainer = DownstreamTrainer()
    trainer.run_finetuning()
