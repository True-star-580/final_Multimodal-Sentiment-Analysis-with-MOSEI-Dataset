import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioSentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super(AudioSentimentModel, self).__init__()
        
        # Define the model architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
    
    def forward(self, x):
        # First fully connected layer with batch normalization and ReLU
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second fully connected layer with batch normalization and ReLU
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

class TransformerAudioEncoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim=128, 
        num_layers=2, 
        num_heads=4, 
        dropout_rate=0.3
    ):
        super(TransformerAudioEncoder, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, 1)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    # Get encoded features without prediction (for multimodal fusion)
    def get_encoded_features(self, x):
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add batch dimension if needed (for transformer)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Extract the encoded representation (use the first token)
        encoded = x[:, 0, :]
        
        return encoded
    
    def forward(self, x):
        # Get encoded features
        encoded = self.get_encoded_features(x)
        
        # Project to sentiment score
        sentiment = self.output_projection(encoded)
        
        return encoded, sentiment