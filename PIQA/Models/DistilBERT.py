import sys

sys.path.append(".")

import torch
from transformers import DistilBertTokenizer, DistilBertModel


class DownstreamModel(torch.nn.Module):
    def __init__(self, vocab_size=31000):
        super(DownstreamModel, self).__init__()
        self.vocab_size = vocab_size
        self.distil = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                      return_dict=True)
        # for param in self.distil.parameters():
        #   param.requires_grad = False

        self.cls_layer = torch.nn.Linear(768, 256)
        self.cls_layer2 = torch.nn.Linear(256, 128)
        self.cls_layer3 = torch.nn.Linear(128, 4)

        self.relu = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

        self.cls_loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, ans, labels):
        ans_cls = self.distil(ans).last_hidden_state[:, 0, :].squeeze(1)
        answer_logits = self.cls_layer(ans_cls)
        answer_logits = self.relu(answer_logits)
        answer_logits = self.cls_layer2(answer_logits)
        answer_logits = self.relu2(answer_logits)
        answer_logits = self.cls_layer3(answer_logits)

        qa_loss = self.qa_loss_f(answer_logits=answer_logits, labels=labels)
        answer_preds = torch.argmax(answer_logits, dim=-1)
        return answer_preds, qa_loss

    def qa_loss_f(self, answer_logits, labels):
        return self.cls_loss_func(input=answer_logits,
                                  target=labels)


