import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DistilBertTokenizer, DistilBertModel


categories = [
    'unb',
    'shal',
    'pres',
    'auth',
    'met',
    'comp',
    'merr'
]


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(categories))
        self.pcl_head = nn.Linear(hidden_dim, 1)  # PCL prediction head
    def forward(self, x):
        x = F.relu(self.fc1(x))
        categories_out = self.fc2(x)
        pcl_out = self.pcl_head(x).squeeze(-1)  # Shape: (batch_size,)
        out = {category: categories_out[:, idx] for idx, category in enumerate(categories)}
        out['pcl'] = pcl_out
        return out


class AttentionClassifier(nn.Module):
    def __init__(self, input_dim: int = 768, num_heads: int = 1, hidden_dim: int = 256):
        super(AttentionClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_categories = len(categories)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        self.logit_layer = nn.Linear(input_dim, 1)
        self.pcl_head = nn.Linear(input_dim, 1)  # PCL prediction head

    def forward(self, x):
        x_tokens = x.unsqueeze(1).repeat(1, self.num_categories, 1)
        
        attn_out, _ = self.attention(x_tokens, x_tokens, x_tokens)
        x_tokens = x_tokens + attn_out
        
        logits = self.logit_layer(x_tokens).squeeze(-1)
        
        out = {category: logits[:, idx] for idx, category in enumerate(categories)}
        
        # PCL prediction from original pooled representation
        pcl_out = self.pcl_head(x).squeeze(-1)  # Shape: (batch_size,)
        out['pcl'] = pcl_out
        return out


class PCLModel(nn.Module):
    def __init__(self, classifier, pretrained_model_name: str = 'distilbert-base-uncased'):
        super(PCLModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        if self.training:
            return logits
        # Apply sigmoid to all predictions including PCL
        result = {cat: torch.sigmoid(logits[cat]) for cat in categories}
        result['pcl'] = torch.sigmoid(logits['pcl'])
        return result


def get_classifier(type: str = 'linear', input_dim: int = 768, hidden_dim: int = 256):
    if type == 'linear':
        return LinearClassifier(input_dim, hidden_dim)
    elif type == 'attention':
        return AttentionClassifier(input_dim, num_heads=1, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported classifier type: {type}")

def get_tokenizer(pretrained_model_name: str = 'distilbert-base-uncased'):
    return DistilBertTokenizer.from_pretrained(pretrained_model_name)

def get_model(classifier_type: str = 'linear', pretrained_model_name: str = 'distilbert-base-uncased', checkpoint_path: str = None):
    classifier = get_classifier(type=classifier_type, input_dim=768, hidden_dim=64)
    model = PCLModel(classifier, pretrained_model_name)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    return model
