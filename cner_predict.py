from transformers import AutoTokenizer,BertModel,BertPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config,):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,end_positions=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type =='lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs

class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x

class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x

def bert_extract_item(start_pos, stop_pos):
	S = []
	start_pred = torch.argmax(start_pos, -1).cpu().numpy()[0][1:-1]
	end_pred = torch.argmax(stop_pos, -1).cpu().numpy()[0][1:-1]
	for i, s_l in enumerate(start_pred):
		if s_l == 0:
			continue
		for j, e_l in enumerate(end_pred[i:]):
			if s_l == e_l:
				S.append((s_l, i, i + j))
				break
	return S

label_list = ['O', 'NUM', 'MEET', 'HELP', 'NAME', 'FUNC', 'JOIN', 'FREE', 'EXIT', 'SIGN', 'OCCU', 'CALL', 'REC', 'ADD', 'FEED']
id2label = {i: label for i, label in enumerate(label_list)}
sequence = "你能回答什么问题"
checkpoint = "./cner"
tokenizer  = AutoTokenizer.from_pretrained(checkpoint, do_lower_case=True)
model = BertSpanForNer.from_pretrained(checkpoint)
hidden_size = model.config.hidden_size
num_labels  = model.config.num_labels
dropout = nn.Dropout(model.config.hidden_dropout_prob)
startlogits = PoolerStartLogits(hidden_size, num_labels)
endlogits = PoolerEndLogits(hidden_size+num_labels, num_labels)

inputs = tokenizer(sequence, padding=True, return_tensors="pt")
model.eval()
with torch.no_grad():
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
start_pos, stop_pos = outputs[:2]
R = bert_extract_item(start_pos, stop_pos)
if R:
	label_entities = [[id2label[x[0]], x[1], x[2]] for x in R]
else:
	label_entities = []
print(f'{label_entities}')
