import sys

sys.path.append(".")
import torch
from SocialIQA.Preprocess.Preprocess import Preprocessor
from SocialIQA.Models.DistilBERT import DownstreamModel
from SocialIQA import constants as con
from torch import optim


class DownstreamTrainer:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.optimizer = None

    def setup_preprocessed_data(self):
        self.preprocessor = Preprocessor()
        self.preprocessor.get_loaders(load_from_pkl=Truex)

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

        self.model.train()
        total_loss = 0
        batch_correct = 0
        total_correct = 0
        index = 0
        for ans, label in train_loader:
            self.optimizer.zero_grad()
            if con.CUDA:
                ans = ans.cuda()
                label = label.cuda()

            answer_preds, qa_loss = self.model(ans, label)
            total_loss += qa_loss
            qa_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            batch_correct += self.evaluate(answer_preds=answer_preds,
                                           labels=label)
            total_correct += con.BATCH_SIZE
            index += 1

            if index % 5 == 0:
                print("Running train loss", qa_loss)
                print("Running train acc", batch_correct / total_correct)

        print("Total train loss:", total_loss / index)
        print("Total train acc:", batch_correct / total_correct)
        print('------------------------')

    def evaluate(self, answer_preds, labels):
        labels = labels.view(-1)
        correct = torch.sum(torch.eq(answer_preds, labels))
        return correct.cpu().detach().numpy()

    def run_pretraining(self):
        self.setup_preprocessed_data()
        self.setup_model()
        self.setup_scheduler_optimizer()
        for epoch in range(30):
            self.train_model()
            # torch.save(self.model.distil.state_dict(), "BERT_retraining/Data/core_model" + str(epoch))
            # torch.save(self.model.state_dict(), "BERT_retraining/Data/mlm_nsp_model" + str(epoch))


if __name__ == '__main__':
    trainer = DownstreamTrainer()
    trainer.run_pretraining()
