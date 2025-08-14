# src/models/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# This base Transformer block is now more robust.
class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # Using norm_first=True is a modern practice for better stability in Transformers
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

class TransformerTextEncoder(TransformerEncoderBlock):
    pass
class TransformerAudioEncoder(TransformerEncoderBlock):
    pass
class TransformerVisualEncoder(TransformerEncoderBlock):
    pass

# The main model bringing it all together
class TransformerFusionModel(nn.Module):
    def __init__(self, text_dim, audio_dim, visual_dim, hidden_dim, num_layers, num_heads, dropout_rate):
        super().__init__()
        
        # Individual encoders for each modality
        self.text_encoder = TransformerTextEncoder(text_dim, hidden_dim, num_layers, num_heads, dropout_rate)
        # For simplicity and stability, we'll focus on a text-centric model first.
        # You can integrate the other encoders later if needed.
        # self.audio_encoder = TransformerAudioEncoder(audio_dim, hidden_dim, 2, num_heads, dropout_rate)
        # self.visual_encoder = TransformerVisualEncoder(visual_dim, hidden_dim, 2, num_heads, dropout_rate)
        
        # Final output layer
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        text_data = inputs.get('language')
        
        # This is the key fix. If the input is missing the sequence dimension, we add it.
        # This handles cases where the sequence length is 1.
        if text_data.dim() == 2:
            text_data = text_data.unsqueeze(1) # Shape becomes [batch, 1, features]

        # Create a mask for padding.
        # Now that we've guaranteed text_data is 3D, this line is safe.
        text_mask = (torch.sum(text_data, dim=2) != 0)

        # Pass data through the text encoder
        # The transformer layer expects the mask to be inverted (True for values to be ignored)
        text_feat = self.text_encoder(text_data, mask=~text_mask)
        
        # Mean pool over the sequence length, ignoring padded elements
        # This is a robust way to get a single vector representation for the whole sequence
        masked_text_features = text_feat * text_mask.unsqueeze(-1)
        pooled_output = torch.sum(masked_text_features, dim=1) / text_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

        # Final prediction
        output = self.output_proj(pooled_output)
        return output
