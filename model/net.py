import torch.nn as nn
from pytorch_pretrained_bert import BertPreTrainedModel, BertModel


class BertTagger(BertPreTrainedModel):
    """
    KorBERT Sequence Tagger
    Args:
        config (BertConfig) : Configs
        num_labels (int) : length of label_vocab
        vocab (Vocab) : token_vocab
    """
    def __init__(self, config, num_labels, vocab):
        super(BertTagger, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.vocab = vocab
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, labels=None):
        attention_mask = input_ids.ne(self.vocab.to_indices(self.vocab.padding_token)).float()
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits