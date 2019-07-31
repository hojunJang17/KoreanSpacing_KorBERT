import json
import torch
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.utils import batchify
from model.data import Corpus
from model.net import BertTagger
from sklearn.metrics import f1_score
from pytorch_pretrained_bert.modeling import BertConfig


def get_eval(model, data_loader, device):
    # Evaluation method : f1 score

    if model.training:
        model.eval()

    true_entities = []
    pred_entities = []

    for mb in tqdm(data_loader, desc='steps'):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)
        y_mb = y_mb.cpu()
        with torch.no_grad():
            yhat = model(x_mb)
            yhat = yhat.max(2)[1].cpu()
            pred_entities.extend(yhat.masked_select(y_mb.ne(0)).numpy().tolist())
            true_entities.extend(y_mb.masked_select(y_mb.ne(0)).numpy().tolist())
    else:
        score = f1_score(true_entities, pred_entities, average='weighted')
    return score


with open('experiment/config.json') as f:
    params = json.loads(f.read())

# loading token & label vocab
token_vocab_path = params['filepath'].get('token_vocab')
label_vocab_path = params['filepath'].get('label_vocab')
with open(token_vocab_path, 'rb') as f:
    token_vocab = pickle.load(f)
with open(label_vocab_path, 'rb') as f:
    label_vocab = pickle.load(f)


# loading trained model
# save_path = params['filepath'].get('ckpt')
save_path = 'tokenize.pth'
ckpt = torch.load(save_path)
config = BertConfig('bert/bert_config.json')
model = BertTagger(config=config, num_labels=len(label_vocab.token_to_idx), vocab=token_vocab)
model.load_state_dict(ckpt['model_state_dict'])

# loading datasets
batch_size = params['training'].get('batch_size')
train_path = 'dataset/new_train.pkl'
val_path = 'dataset/new_val.pkl'
test_path = 'dataset/new_test.pkl'

train_data = Corpus(train_path, token_vocab.to_indices, label_vocab.to_indices)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16,
                          drop_last=True, collate_fn=batchify)
val_data = Corpus(val_path, token_vocab.to_indices, label_vocab.to_indices)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=16,
                        drop_last=True, collate_fn=batchify)
test_data = Corpus(test_path, token_vocab.to_indices, label_vocab.to_indices)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16,
                         drop_last=True, collate_fn=batchify)


# using gpu
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# evaluating
print('Evaluating train set')
train_f1_score = get_eval(model, train_loader, device)

print('Evaluating validation set')
val_f1_score = get_eval(model, val_loader, device)

print('Evaluating test set')
test_f1_score = get_eval(model, test_loader, device)

print('Train Set f1_score: {:.2%}'.format(train_f1_score))
print('Validation Set f1_score: {:.2%}'.format(val_f1_score))
print('Test Set f1_score: {:.2%}'.format(test_f1_score))