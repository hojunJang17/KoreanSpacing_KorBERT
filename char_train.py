import json
import pickle
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_pretrained_bert.modeling import BertConfig
from model.data import Corpus
from model.net import BertTagger
from model.utils import batchify



def evaluate(model, data_loader, device):
    if model.training:
        model.eval()

    model.eval()
    avg_loss = 0
    for step, mb in tqdm(enumerate(data_loader), desc='eval_step', total=len(data_loader)):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            mb_loss = model(x_mb, labels=y_mb)
        avg_loss += mb_loss.item()
        x_mb.cpu()
        y_mb.cpu()
    else:
        avg_loss /= (step+1)

    return avg_loss


# load configs
with open('experiment/config.json') as f:
    params = json.loads(f.read())

# loading vocabs
token_vocab_path = params['filepath'].get('token_vocab')
label_vocab_path = params['filepath'].get('label_vocab')
with open(token_vocab_path, 'rb') as f:
    token_vocab = pickle.load(f)
with open(label_vocab_path, 'rb') as f:
    label_vocab = pickle.load(f)


# model
config = BertConfig('bert/bert_config.json')
model = BertTagger(config=config, num_labels=len(label_vocab.token_to_idx), vocab=token_vocab)
bert_pretrained = torch.load('bert/pytorch_model.bin')
model.load_state_dict(bert_pretrained, strict=False)

# training params
epochs = params['training'].get('epochs')
batch_size = params['training'].get('batch_size')
learning_rate = params['training'].get('learning_rate')
summary_step = params['training'].get('summary_step')


# creating dataset, dataloader
# train_path = params['filepath'].get('train')
# val_path = params['filepath'].get('val')
train_path = 'dataset/new_train.pkl'
val_path = 'dataset/new_val.pkl'
train_data = Corpus(train_path, token_vocab.to_indices, label_vocab.to_indices)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16,
                          drop_last=True, collate_fn=batchify)
val_data = Corpus(val_path, token_vocab.to_indices, label_vocab.to_indices)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=16,
                        drop_last=True, collate_fn=batchify)


# optimizer
opt = optim.Adam([
    {"params": model.bert.parameters(), "lr": learning_rate/100},
    {"params": model.classifier.parameters(), "lr": learning_rate}
])

# using gpu
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# train
for epoch in tqdm(range(epochs), desc='epochs'):
    tr_loss = 0
    model.train()
    for step, mb in tqdm(enumerate(train_loader), desc='train_steps', total=len(train_loader)):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)
        opt.zero_grad()
        mb_loss = model(x_mb, labels=y_mb)

        mb_loss.backward()
        opt.step()

        tr_loss += mb_loss.item()
        x_mb.cpu()
        y_mb.cpu()
        if (epoch * len(train_loader) + step) % summary_step == 0:
            val_loss = evaluate(model, val_loader, device)
            model.train()

    else:
        tr_loss /= (step+1)

    val_loss = evaluate(model, val_loader, device)

    tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch+1, tr_loss, val_loss))

ckpt = {'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict()}
# save_path = params['filepath'].get('ckpt')
save_path = 'tokenize.pth'
torch.save(ckpt, save_path)
