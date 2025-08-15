# src/models/fusion_v2.py
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return x

class MultilingualSentimentClassifier(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_classes=5, dropout_rate=0.3):
        super().__init__()
        self.bert = bert_model
        
        # Simple classifier on top of BERT's [CLS] token output
        self.classifier = nn.Sequential(
            nn.Linear(bert_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    # --- AMENDED THIS LINE ---
    # We now accept `token_type_ids` and a general `**kwargs` to gracefully
    # ignore any extra arguments passed by the tokenizer.
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
    # --- END AMENDMENT ---
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token's representation for classification
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits