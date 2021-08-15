import re
from tqdm import tqdm
from transformers import BertConfig, BertForTokenClassification, get_linear_schedule_with_warmup
from transformers import BertTokenizer
import json
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.optim import Adam
from torch import nn
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c',"--cuda",type = int)
parser.add_argument('-d',"--depict",type = str)
parser.add_argument('-b',"--batch_size",type = int)
parser.add_argument('-m',"--maxlen",type = int)
parser.add_argument('-lr',"--learning_rate",type = float)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.cuda)
batch_size = 32
epochs = 10
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
#scp -P31415 -r /Users/lijunlin/Documents/Self/yan22/bertrefine/ontonotes suqi@162.105.23.35:~/bertpretrain
def process_single(data):
    rows = data.strip().split("\n")
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

def tokenizer(output):
    token_ids = []
    type_ids = []
    sequence = zip(output[0], output[1])
    for token,type in type_ids:
        token_id = tokenizer.encode(token,add_special_tokens=False)
        type_id = [type] * len(token_id)

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
    map = {}
    map['O'] = 0
    map['[CLS]'] = 1
    map['[SEP]'] = 2
    map['B_V'] = 3
    map['I_V'] = 4
    for i,x in enumerate(label_set): 
        if not x in map.keys():
            map[x] = i + 5
    return map

def read_prepared_data(path,maxlen):
    f = open(path,"r").read()
    datas = f.split("\n\n\n")[:-1]
    input_ids = []
    labels = []
    tokens = []
    attns = []
    for data in tqdm(datas):
        input_id = [101]
        label = ['[CLS]']
        token = ['[CLS]']
        attn = [1]
        seqs = data.split("\n")
        for seq in seqs:
            id, role, word = seq.split(" -- ")
            input_id.append(int(id))
            label.append(role)
            token.append(word)
            attn.append(1)
        input_id.append(102)
        label.append('[SEP]')
        token.append('[SEP]')
        attn.append(1)
        while len(input_id) < maxlen:
            input_id.append(0)
            label.append("O")
            token.append("[PAD]")
            attn.append(0)
        if len(input_id) > maxlen:
            input_id = input_id[:maxlen]
            label = label[:maxlen]
            token = token[:maxlen]
            attn = attn[:maxlen]
        assert len(input_id) == maxlen
        assert len(label) == maxlen
        assert len(token) == maxlen
        assert len(attn) == maxlen
        input_ids.append(input_id)
        labels.append(label)
        tokens.append(token)
        attns.append(attn)
    return input_ids, attns, labels, tokens

def data_loading(input_ids, attns, labels, map):
    input_ids = torch.LongTensor(input_ids)
    attns =  torch.LongTensor(attns)
    mapping = lambda x:[map[y] for y in x]
    labels = [mapping(x) for x in labels]
    labels = torch.LongTensor(labels)
    dataset = TensorDataset(input_ids, attns, labels)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    return dataset, dataloader

def get_acc(outputs,labels,attns):
    outputs = outputs.detach().cpu()
    labels = labels.detach().cpu()
    attns = attns.detach().cpu()
    target = labels[attns == 1]
    predict = outputs[attns == 1]
    questioned = 0
    for i in range(outputs.size(0)):
        if outputs[i,:][attns[i,:] == 1][1:-1].sum() == 0:
            questioned +=1
    if questioned == outputs.size(0):
        print("警告，全部预测为O")
    acc = (target == predict).float().mean()
    return acc.item()

def initial_scheduler(dataloader,epochs,warmup_proportion,optimizer):
    num_training_steps = len(dataloader) * epochs
    num_warmup_steps = num_training_steps * warmup_proportion
    #last_epoch = (last_epoch_id+1)*len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return scheduler

class SRL(nn.Module):
    def __init__(self, model, lr):
        super(SRL,self).__init__()
        self.model = model
        self.model = self.model.cuda()
        self.num_class = num_class
        self.lr = lr
        self.optimizer = Adam(self.parameters(),lr = self.lr)
    def forward(self, input_ids, attn_ids, targets = None):
        outputs = self.model(input_ids = input_ids, attention_mask = attn_ids, labels = targets)
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits
    def trainer(self,dataloader):
        self.train()
        current_loss = 0
        current_acc = 0
        with tqdm(enumerate(dataloader)) as t:
            t.set_description("total_iter = {}".format(len(dataloader)))
            for i, (inputs, attns, labels) in t:
                #mask = (labels>=0).byte()
                inputs = inputs.cuda()
                attns = attns.cuda()
                labels = labels.cuda()
                #mask = mask.cuda()
                loss, logits = self.forward(input_ids = inputs,
                                            attn_ids = attns,
                                            targets = labels)
                output = logits.argmax(dim = -1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                current_loss += (loss.item() - current_loss)/(i+1)
                with torch.no_grad():
                    acc = get_acc(output, labels ,attns)
                current_acc += (acc - current_acc)/(i+1) 
                t.set_postfix(epoch = i+1, current_acc = current_acc,current_loss = current_loss)
                t.update()
        return current_loss, current_acc
    def tester(self,dataloader):
        self.eval()
        current_loss = 0
        current_acc = 0
        with tqdm(enumerate(dataloader)) as t:
            t.set_description("total_iter = {}".format(len(dataloader)))
            for i, (inputs, attns, labels) in t:
                #mask = (labels>=0).byte()
                inputs = inputs.cuda()
                attns = attns.cuda()
                labels = labels.cuda()
                #mask = mask.cuda()
                with torch.no_grad():
                    loss, logits = self.forward(input_ids = inputs,
                                                attn_ids = attns,
                                                targets = labels)
                    #torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    output = logits.argmax(dim = -1)
                    acc = get_acc(output, labels ,attns)
                    current_loss += (loss.item() - current_loss)/(i+1)
                    current_acc += (acc - current_acc)/(i+1) 
                t.set_postfix(epoch = i+1, current_acc = current_acc,current_loss = current_loss)
                t.update()
        return current_loss, current_acc

def main(model,train_loader,dev_loader,depict,round):
    srl = SRL(model, lr = lr)       
    srl = srl.cuda()
    history = {x:[] for x in ['lr',"train_loss","train_acc","dev_loss","dev_acc","best_acc"]}
    history['lr'] = srl.lr
    best_acc = 0
    step = 0
    patience =7
    if "best_model" not in os.listdir():
        os.mkdir("best_model")
    if "{}_{}".format(depict,round) not in os.listdir("best_model"):
        os.mkdir("best_model/{}_{}/".format(depict,round))
    output_model_file = "best_model/{}_{}/model_file.bin".format(depict,round)
    output_config_file = "best_model/{}_{}/config_file.bin".format(depict,round)
    output_vocab_file = "best_model/{}_{}/vocab_file.bin".format(depict,round)
    with tqdm(range(epochs)) as t:
        for epoch in t:
            train_loss, train_acc = srl.trainer(train_loader)
            dev_loss, dev_acc = srl.tester(dev_loader)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["dev_loss"].append(dev_loss)
            history["dev_acc"].append(dev_acc)
            if dev_acc >= best_acc:
                best_acc = dev_acc
                history['best_acc'] = best_acc
                step = 0
                model_to_save = srl.model.module if hasattr(srl, 'module') else srl.model  ##刷新模型
            else:
                if epoch > 3:
                    step += 1
                else:
                    pass
            t.set_postfix(
                dev_acc = dev_acc,
                best_acc = best_acc)
            if step <= patience:
                pass
            else:
                print("best_acc={}".format(best_acc))
                break        
    if "srl_result" not in os.listdir():
        os.mkdir("srl_result")
    to_dump = json.dumps(history)
    f = open("srl_result/{}".format(depict),"a+")
    json.dump(to_dump, f)
    f.write("\n")
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_vocab_file)
    return best_acc

def load_my_model(start_epoch):
    output_model_file = "/home/suqi/bertpretrain/with_pos/epoch{}/model_file.bin".format(start_epoch)
    output_config_file = "/home/suqi/bertpretrain/with_pos/epoch{}/config_file.bin".format(start_epoch)
    output_vocab_file = "/home/suqi/bertpretrain/with_pos/epoch{}/vocab_file.bin".format(start_epoch)
    model_state_dict = torch.load(output_model_file)
    config = BertConfig.from_json_file(output_config_file)
    config.num_labels = num_class
    model = BertForTokenClassification.from_pretrained("bashar-talafha/multi-dialect-bert-base-arabic",state_dict=model_state_dict,config = config)
    tokenizer = BertTokenizer(output_vocab_file)
    return tokenizer,model



if __name__ == "__main__":
    batch_size = args.batch_size
    maxlen = args.maxlen
    lr = args.lr
    depict = args.depict
    train_input_ids, train_attns, train_labels, train_tokens = read_prepared_data("bert_tokenized_train.txt",maxlen = maxlen)
    dev_input_ids, dev_attns, dev_labels, dev_tokens = read_prepared_data("bert_tokenized_dev.txt",maxlen = maxlen)
    if "label_map.json" not in os.listdir():
        label_map = get_label_set(train_labels+dev_labels)
        with open("label_map.json","w+") as map:
            json.dump(label_map, map)
    else:
        with open("label_map.json","r") as map:
            label_map = json.load(map)
    num_class = max(label_map.values()) +1
    model = BertForTokenClassification.from_pretrained('asafaya/bert-base-arabic', num_labels = num_class)
    tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
    #_, bert = load_model()
    train_set, train_loader = data_loading(train_input_ids, train_attns, train_labels, label_map)
    dev_set, dev_loader = data_loading(dev_input_ids, dev_attns, dev_labels, label_map)
    #best_acc_bert = main(bert,"bert_5_3",train_loader,dev_loader)
    best_acc = main(model,train_loader,dev_loader,depict,round)