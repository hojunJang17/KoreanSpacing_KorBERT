import json
import pickle
from tqdm import tqdm
from glob import glob
from bert.tokenization import BertTokenizer
from sklearn.model_selection import train_test_split

with open('experiment/config.json') as f:
    params = json.loads(f.read())

filenames = glob('sentences/*.txt')
ptr_tokenizer = BertTokenizer.from_pretrained('bert/vocab.korean.rawtext.list', do_lower_case=False)

data = []

for p in tqdm(filenames):
    with open(p, 'r', encoding='utf-8') as f:
        a = f.readlines()
        for aaa in range(len(a)):
            b = a[aaa].split()
            if not b:
                continue
            label = ['<split>']*(len(b)-1)
            token = ptr_tokenizer.tokenize(a[aaa].replace(' ', ''))
            index = 0
            for i in range(len(b)):
                k = ptr_tokenizer.tokenize(b[i])
                count = 0
                while count < len(k)-1:
                    label.insert(index, '<non_split>')
                    index += 1
                    count += 1
                index += 1
            label.append('<non_split>')
            if len(token) != len(label):
                continue
            data.append([token, label])


train_data, test_data = train_test_split(data, test_size=0.2, random_state=77)
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=7)

# saving data
train_path = params['filepath'].get('train')
val_path = params['filepath'].get('val')
test_path = params['filepath'].get('test')

with open(train_path, 'wb') as f:
    pickle.dump(train_data, f)
with open(val_path, 'wb') as f:
    pickle.dump(val_data, f)
with open(test_path, 'wb') as f:
    pickle.dump(test_data, f)