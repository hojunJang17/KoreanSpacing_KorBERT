import json
import pickle
from model.utils import Vocab
from bert.tokenization import BertTokenizer


with open('experiment/config.json') as f:
    params = json.loads(f.read())

# loading BertTokenizer
ptr_tokenizer = BertTokenizer.from_pretrained('bert/vocab.korean.rawtext.list', do_lower_case=False)
idx_to_token = list(ptr_tokenizer.vocab.keys())

# generate vocab
token_vocab = Vocab(idx_to_token, padding_token='[PAD]', unknown_token='[UNK]', bos_token=None,
              eos_token=None, reserved_tokens=['[CLS]', '[SEP]', '[MASK]'], unknown_token_idx=1)
label_vocab = Vocab(['<split>', '<non_split>'], unknown_token=None, bos_token=None, eos_token=None)


# save vocab
token_vocab_path = params['filepath'].get('token_vocab')
label_vocab_path = params['filepath'].get('label_vocab')
with open(token_vocab_path, 'wb') as f:
    pickle.dump(token_vocab, f)
with open(label_vocab_path, 'wb') as f:
    pickle.dump(label_vocab, f)