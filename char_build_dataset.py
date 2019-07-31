import json
import pickle
import re
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split


def get_label(s, idx=0):
    # labeling data

    label = []
    while True:
        try:
            next_ch = s[idx+1]
        except:
            label.append('<non_split>')
            break
        if next_ch == ' ':
            label.append('<split>')
            while s[idx+1] == ' ':
                idx += 1
            idx += 1
        else:
            label.append('<non_split>')
            idx += 1
    return label

with open('experiment/config.json') as f:
    params = json.loads(f.read())

filenames = glob('sentences/*.txt')

data = []
for p in tqdm(filenames):
    treg = re.compile('<\w*>|</\w*>')
    wreg = re.compile(' ')
    with open(p, 'r', encoding='utf-8') as f:
        a = f.readlines()
        a = [treg.sub('', str(t)).strip() for t in a]
        # a = [t.lower() for t in a if t]
        labels = [get_label(t) for t in a]
        a = [wreg.sub('', str(t)) for t in a]
        for i in range(len(a)):
            if len(a[i]) < 512:
                # bert max_position_embeddings = 512
                data.append(([[char for char in a[i]], labels[i]]))


train_data, test_data = train_test_split(data, test_size=0.2, random_state=77)
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=7)

# saving data
with open('dataset/new_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('dataset/new_val.pkl', 'wb') as f:
    pickle.dump(val_data, f)
with open('dataset/new_test.pkl', 'wb') as f:
    pickle.dump(test_data, f)