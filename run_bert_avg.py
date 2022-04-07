import argparse
import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel, BertConfig

from data_utils import load_STS_data, TestDataset

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="model_hub/hfl_chinese-roberta-wwm-ext/")
parser.add_argument('--max_len', type=int, default=64)
parser.add_argument('--pooling', type=str, default="cls")
parser.add_argument('--batch_size', type=int, default=32)

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained(args.model_path)

dev_data = load_STS_data("./data/STS-B/cnsd-sts-dev.txt")
test_data = load_STS_data("./data/STS-B/cnsd-sts-test.txt")


class model(nn.Module):
    def __init__(self, model_path, ):
        super(model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           output_hidden_states=True)
        if args.pooling == "cls":
            output = output[0][:, -1, :]
        elif args.pooling == "first_last":
            hidden_states = output.hidden_states
            output = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        return output


def test(test_data, model, batch_size):
    target_idxs, source_idxs, label_list = test_data.get_data()
    total_step = target_idxs['input_ids'].shape[0] // batch_size \
        if target_idxs['input_ids'].shape[0] % batch_size else target_idxs['input_ids'].shape[0] // batch_size + 1

    label_list_all = []
    similarity_list_all = []
    with torch.no_grad():
        for i in range(total_step):
            start = i * batch_size
            end = (i+1) * batch_size
            target_input_ids = target_idxs['input_ids'][start:end].to(device)
            target_attention_mask = target_idxs['attention_mask'][start:end].to(device)
            target_token_type_ids = target_idxs['token_type_ids'][start:end].to(device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)

            source_input_ids = source_idxs['input_ids'][start:end].to(device)
            source_attention_mask = source_idxs['attention_mask'][start:end].to(device)
            source_token_type_ids = source_idxs['token_type_ids'][start:end].to(device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)

            similarity_list = F.cosine_similarity(target_pred, source_pred)
            similarity_list = similarity_list.cpu().numpy()

            similarity_list_all.extend(similarity_list)
            label_list_batch = np.array(label_list[start:end])
            label_list_all.extend(label_list_batch)

        corrcoef = scipy.stats.spearmanr(label_list_all, similarity_list_all).correlation
    return corrcoef


if __name__ == '__main__':
    model = model(args.model_path).to(device)
    deving_data = TestDataset(dev_data, tokenizer, args.max_len)
    testing_data = TestDataset(test_data, tokenizer, args.max_len)
    corrcoef = test(deving_data, model, args.batch_size)
    print("dev corrcoef: {}".format(corrcoef))
    corrcoef = test(testing_data, model, args.batch_size)
    print("test corrcoef: {}".format(corrcoef))