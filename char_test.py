import json
import torch
import pickle
import re
from model.net import BertTagger
from pytorch_pretrained_bert.modeling import BertConfig

def result(x, y):
    """
    takes x, y and converts into string
    """
    A = ''
    for i in range(len(x)):
        A += x[i]
        if y[i] == '<split>':
            A += ' '
    return A


with open('experiment/config.json') as f:
    params = json.loads(f.read())

# ptr_tokenizer = BertTokenizer.from_pretrained('bert/vocab.korean.rawtext.list', do_lower_case=False)

# loading vocabs
token_vocab_path = params['filepath'].get('token_vocab')
label_vocab_path = params['filepath'].get('label_vocab')
with open(token_vocab_path, 'rb') as f:
    token_vocab = pickle.load(f)
with open(label_vocab_path, 'rb') as f:
    label_vocab = pickle.load(f)


# loading trained model
save_path = params['filepath'].get('ckpt')
# save_path= 'tokenize.pth'
ckpt = torch.load(save_path)
config = BertConfig('bert/bert_config.json')
model = BertTagger(config=config, num_labels=len(label_vocab.token_to_idx), vocab=token_vocab)
model.load_state_dict(ckpt['model_state_dict'])

while True:
    original_text = input("Input:\t")
    if original_text == '':
        print('exit')
        break
    treg = re.compile('<\w*>|</\w*>')
    wreg = re.compile(' ')
    a = [treg.sub('', str(t)).strip() for t in original_text]
    a = [t for t in a if t]
    a = [wreg.sub('', str(t)) for t in a]
    b = torch.tensor(token_vocab.to_indices(a), dtype=torch.int64)
    b = b.view(1, -1)
    with torch.no_grad():
        yhat = model(b).max(2)[1].numpy().tolist()
        print('Output:\t', result(a, label_vocab.to_tokens(yhat[0])), '\n')