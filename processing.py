import re
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

def process_single(data):
    rows = data.strip().split("")
    rows = [x.split() for x in rows]
    ##token = row[3]
    ##pradicate = row[10:len(row)-1]
    tokens = [row[3] for row in rows]
    type_rows = [x for x in range(11,len(rows[0])-1)]
    types = []
    for type_row in type_rows:
        type = [row[type_row] for row in rows]
        types.append(type)
    output = []
    for type in types:
        tokens_with_indicator = []
        label_with_indicator = []
        for token,label in zip(tokens, type):
            if label == "(V*)":
                tokens_with_indicator.extend(["$",token,"$"])
                label_with_indicator.extend(["*",label,"*"])
            else:
                tokens_with_indicator.append(token)
                label_with_indicator.append(label)
        output.append((tokens_with_indicator,label_with_indicator))
    return output

def process_all(datas):
    outputs = []
    for data in tqdm(datas):
        output = process_single(data)
        outputs += output
    return outputs

def type_converter(seq):
    pattern = re.compile(r"(?<=\().*?(?=\*)")
    current = ["O"]
    result = ["O"] * len(seq)
    extra_bracket = 0
    for i in range(len(seq)):
        if pattern.search(seq[i]):
            current_label = pattern.search(seq[i]).group().split("(")[0]
        else:
            current_label = "O"
        if extra_bracket>0:
            result[i] = result[i-1]
        if extra_bracket == 0:
            if seq[i] == "*":
                pass
            else:
                result[i] = current_label
        left, right = count_bracket(seq[i])
        extra_bracket = extra_bracket + left - right
    gold_result = []
    for i in range(len(result)):
        if result[i] == "O":
            gold_result.append(result[i])
        else:
            if i == 0 or result[i] != result[i-1]:
                gold_result.append("B_" + result[i])
            else:
                gold_result.append("I_" + result[i])
    return gold_result
def count_bracket(string):
    left = 0
    right = 0
    for char in string:
        if char == "(":
            left += 1
        if char == ")":
            right += 1
    return left,right

def tokenizer_with_type(tokens, types):
    ids = []
    types_new = []
    step = 0
    for token in tokens:
        token = token.lower()
        id = tokenizer.encode(token, add_special_tokens=False)
        ids += id
        if types[step][0] == "I" or types[step][0] == "O":
            types_new_to_add = [types[step]] * len(id)
            types_new += types_new_to_add
        if types[step][0] == "B":
            behind = "I" + types[step][1:]
            types_new_to_add = [types[step]] + [behind] * (len(id)-1)
            types_new += types_new_to_add
        step += 1
    tokens_new = tokenizer.convert_ids_to_tokens(ids)
    return ids, types_new, tokens_new

def preprocess(path, write_path):
    f = open(path,"r").read()
    datas = f.split("\n\n")
    datas = [x for x in datas if not x == '']
    processed_data = process_all(datas)
    full_data = [(data[0],type_converter(data[1]),data[1]) for data in processed_data]
    ids_all = []
    labels_all = []
    tokens_all = []
    with open(write_path,"w+") as file:
        for data in tqdm(full_data):
            tokens, types, _ = data
            ids_new, types_new, tokens_new = tokenizer_with_type(tokens, types)
            ids_all.append(ids_new)
            labels_all.append(types_new)
            tokens_all.append(tokens_new)
            size = len(ids_new)
            for i in range(size):
                to_write = "{} -- {} -- {}\n".format(ids_new[i], types_new[i],tokens_new[i])
                file.write(to_write)
            file.write("\n\n")
    return ids_all, labels_all, tokens_all

def get_label_set(labels):
    all_labels = []
    for label in labels:
        all_labels += label
    label_set = list(set(all_labels))
    map = {x:i for i,x  in enumerate(label_set)}
    map['PAD'] = -1
    map['CLS'] = -1
    map['SEP'] = -1
    return map

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
    train_path = "/english/train.txt"
    dev_path = "/english/dev.txt"
    test_path = "/english/test.txt"
    train_ids, train_labels, train_tokens = preprocess(train_path, "bert_tokenized_train.txt")
    train_ids, train_labels, train_tokens = preprocess(dev_path, "bert_tokenized_dev.txt")
    train_ids, train_labels, train_tokens = preprocess(test_path, "bert_tokenized_test.txt")