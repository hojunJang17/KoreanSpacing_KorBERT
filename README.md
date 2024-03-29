# Korean Spacing (KorBERT)

## Data set

* Sejong Corpus
    * Collection of corpora of modern Korean, International Korean, old Korean and oral folklore literature
    * Link: <https://ithub.korean.go.kr/user/guide/corpus/guide1.do>


## Before training & evaluating

Data size of pickles which were saving data and vocabs were too big to upload in my git repository.
1. Make "./dataset" directory for saving data and vocabs.
2. Run "build_dataset.py", "build_vocab.py"

Need "bert/pytorch_model.bin" to load pretrained net.

Need to run train to save trained model.

## Result

* With BERT Tokenizer
    * Train_f1_score = 98.64%
    * Val_f1_score = 98.37%
    * Test_f1_score = 98.39%

* With Character-level Tokenizer
    * Train_f1_score = 98.78%
    * Val_f1_score = 98.78%
    * Test_f1_score = 98.79%
