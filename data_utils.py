import json
import os


class TestDataset():
    def __init__(self, data, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.traget_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
        assert len(self.traget_idxs['input_ids']) == len(self.source_idxs['input_ids'])

    def text_to_id(self, source):
        sample = self.tokenizer(source, max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def get_data(self):
        return self.traget_idxs, self.source_idxs, self.label_list


def load_snli_vocab(path):
    data = []
    with open(path) as f:
        for i in f:
            data.append(json.loads(i))
    return data


def load_snli_jsonl(path):
    data = []
    with open(path) as f:
        for i in f:
            data.append(json.loads(i))
    return data


def load_STS_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i in f:
            d = i.split("||")
            sentence1 = d[1]
            sentence2 = d[2]
            score = int(d[3])
            data.append([sentence1, sentence2, score])
    return data

def snli_process():
    """这里主要是构建句子三元组"""
    file_path = "./data/SNLI/"
    train_file = 'cnsd_snli_v1.0.train.jsonl'
    test_file = 'cnsd_snli_v1.0.test.jsonl'
    dev_file = 'cnsd_snli_v1.0.dev.jsonl'
    def data_porcess(path):
        data_entailment = []
        data_contradiction = []
        with open(path, "r+", encoding="utf8") as f:
            lines = f.readlines()
            for item in lines:
                item = eval(item)
                if item['gold_label'] == 'entailment':
                    data_entailment.append(item)
                elif item['gold_label'] == 'contradiction':
                    data_contradiction.append(item)
        # 蕴含
        data_entailment = sorted(data_entailment, key=lambda x: x['sentence1'])
        # 矛盾
        data_contradiction = sorted(data_contradiction, key=lambda x: x['sentence1'])
        process = []
        i = 0
        j = 0
        while i < len(data_entailment):
            origin = data_entailment[i]['sentence1']
            for index in range(j, len(data_contradiction)):
                if data_entailment[i]['sentence1'] == data_contradiction[index]['sentence1']:
                    process.append({'origin': origin, 'entailment': data_entailment[i]['sentence2'],
                                    'contradiction': data_contradiction[index]['sentence2']})
                    j = index + 1
                    break
            while i < len(data_entailment) and data_entailment[i]['sentence1'] == origin:
                i += 1
            print(i)
        with open(path[:-6] + 'proceed.txt', 'w') as f:
            for d in process:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    data_porcess(os.path.join(file_path, train_file))
    data_porcess(os.path.join(file_path, test_file))
    data_porcess(os.path.join(file_path, dev_file))


if __name__ == '__main__':
    # path = 'data/STS-B/cnsd-sts-train.txt'
    # data = load_STS_data(path)
    # print(data)
    snli_process()