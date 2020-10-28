import sys

sys.path.append(".")

import torch
from transformers import DistilBertTokenizer, BertModel


class DownstreamModel(torch.nn.Module):
    def __init__(self, vocab_size=31000, load_path=None):
        super(DownstreamModel, self).__init__()
        self.vocab_size = vocab_size
        self.distil = BertModel.from_pretrained('bert-base-uncased',
                                                return_dict=True)

        if load_path is not None:
            print("Loading model", load_path)
            self.distil.load_state_dict(torch.load(load_path))

        # for param in self.distil.parameters():
        #   param.requires_grad = False

        self.cls_layer = torch.nn.Linear(768 * 3, 768)
        self.cls_layer2 = torch.nn.Linear(768, 128)
        self.cls_layer3 = torch.nn.Linear(128, 4)

        self.relu = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

        self.cls_loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, ans_a, ans_b, ans_c, ans_d, ans_e,
                ans_a_att, ans_b_att, ans_c_att, ans_d_att, ans_e_att, labels):
        ans_a = self.answers(ans_a, ans_a_att)
        ans_b = self.answers(ans_b, ans_b_att)
        ans_c = self.answers(ans_c, ans_c_att)
        ans_d = self.answers(ans_d, ans_d_att)
        ans_e = self.answers(ans_e, ans_e_att)

        ans_cls = torch.cat([ans_a, ans_b, ans_c, ans_d, ans_e], dim=-1)
        answer_logits = self.cls_layer(ans_cls)
        answer_logits = self.relu(answer_logits)
        answer_logits = self.cls_layer2(answer_logits)
        answer_logits = self.relu2(answer_logits)
        answer_logits = self.cls_layer3(answer_logits)

        qa_loss = self.qa_loss_f(answer_logits=answer_logits, labels=labels)
        answer_preds = torch.argmax(answer_logits, dim=-1)
        return answer_preds, qa_loss

    def answers(self, ans, att):
        ans_cls = self.distil(input_ids=ans, attention_mask=att).last_hidden_state[:, 0, :].squeeze(1)
        return ans_cls

    def qa_loss_f(self, answer_logits, labels):
        return self.cls_loss_func(input=answer_logits,
                                  target=labels)
