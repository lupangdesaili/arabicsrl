import re
from tqdm import tqdm
from transformers import BertConfig, BertForTokenClassification
from transformers import BertTokenizer
import json
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.optim import Adam
from torch import nn
import os
import json
import argparse
label_map_path = "/home/suqi/bertpretrain/arabic_ontonotes/label_map.json"


def load_best_model(path):
    output_model_file = path + "/model_file.bin"
    output_config_file = path + "/config_file.bin"
    output_vocab_file = path + "/vocab_file.bin"
    model_state_dict = torch.load(output_model_file)
    config = BertConfig.from_json_file(output_config_file)
    model = BertForTokenClassification.from_pretrained("bashar-talafha/multi-dialect-bert-base-arabic",state_dict=model_state_dict,config = config,map_location=torch.device('cpu'))
    tokenizer = BertTokenizer(output_vocab_file)
    return tokenizer,model

with open(label_map_path,"r") as map:
    label_map = json.load(map)
    label_map = {k:v for v,k in label_map.items()}

tokenizer,model = load_best_model('arabic_2')
input = """هل يمكن الكشف ذاتيا على نفسك لتحدد إن كنت مصابا بفيروس كورونا؟ وهل من طرق الكشف الذاتي حبس التنفس 10 ثوان؟ """
verb = "يمكن"
#input = """أعلنت السلطات في نيجيريا -أمس الجمعة- بدء عملية ل إنقاذ 317 طالبة ثانوية خطف هن مسلحون، بينما سجلت عمليات قتل جديدة في منجم ل الذهب، وسط اتساع دائرة العنف و الفوضى في مناطق مختلفة من البلاد."""
#verb = "أعلنت"
input = input.replace(verb,"${}$".format(verb))
inputs = tokenizer(input,return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
output = model(*[v for k,v in inputs.items()])
labels = output.logits.argmax(dim = -1)
labels = [label_map[x] for x in labels.tolist()[0]]
result = zip(tokens, labels)
arranged_tokens = []
arranged_sr = []
for x,y in zip(tokens, labels):
    if x in ["$","[CLS]","[SEP]"]:
        continue
    if x[0] == "#":
        arranged_tokens[-1] = arranged_tokens[-1] + x.replace("#","")
    else:
        arranged_tokens.append(x)
        if y == "O":
            arranged_sr.append(y)
        else:
            arranged_sr.append(y[2:])

for x,y in zip(arranged_tokens, arranged_sr):
    print("{}     {}".format(x,y))
