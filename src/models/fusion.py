import torch
import torch.nn as nn
import torch.nn.functional as F

# Early fusion model for multimodal sentiment analysis.
# Concatenates features from different modalities and applies fully connected layers.
class EarlyFusionModel(nn.Module):
    def __init__(
        self, 
        text_dim, 
        audio_dim, 
        visual_dim, 
        hidden_dim=256, 
        dropout_rate=0.3
    ):
        super(EarlyFusionModel, self).__init__()
        
        # Calculate total input dimension
        total_dim = text_dim + audio_dim + visual_dim
        
        # Define the fusion architecture
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_features, audio_features, visual_features):
        # Concatenate features
        concat_features = torch.cat([text_features, audio_features, visual_features], dim=1)
        
        # Apply fusion layers
        sentiment = self.fusion_layers(concat_features)
        
        return sentiment

# Late fusion model for multimodal sentiment analysis.
# Applies separate models to each modality and combines the predictions.
class LateFusionModel(nn.Module):
    def __init__(
        self, 
        text_model, 
        audio_model, 
        visual_model, 
        fusion_weights=None
    ):
        super(LateFusionModel, self).__init__()
        
        # Store individual models
        self.text_model = text_model
        self.audio_model = audio_model
        self.visual_model = visual_model
        
        # Freeze individual models
        for model in [self.text_model, self.audio_model, self.visual_model]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Initialize fusion weights
        if fusion_weights is None:
            # Learn the weights if not provided
            self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            # Use fixed weights if provided
            self.fusion_weights = torch.tensor(fusion_weights)
    
    def forward(self, text_features, audio_features, visual_features):
        # Get predictions from each modality
        text_pred = self.text_model(text_features)
        audio_pred = self.audio_model(audio_features)
        visual_pred = self.visual_model(visual_features)
        
        # Combine predictions with weights
        weights = F.softmax(self.fusion_weights, dim=0)
        combined_pred = (
            weights[0] * text_pred + 
            weights[1] * audio_pred + 
            weights[2] * visual_pred
        )
        
        return combined_pred

# Transformer-based fusion model for multimodal sentiment analysis.
# Uses transformer encoders for each modality and cross-attention for fusion.
class TransformerFusionModel(nn.Module):
    def __init__(
        self, 
        text_dim, 
        audio_dim, 
        visual_dim, 
        hidden_dim=256, 
        num_heads=8, 
        num_layers=4, 
        dropout_rate=0.3
    ):
        super(TransformerFusionModel, self).__init__()
        
        # Import here to avoid circular imports
        from src.models.text import TransformerTextEncoder
        from src.models.audio import TransformerAudioEncoder
        from src.models.visual import TransformerVisualEncoder
        from src.models.attention import MultimodalCrossAttention
        
        # Individual modality encoders
        self.text_encoder = TransformerTextEncoder(
            text_dim, hidden_dim, num_layers, num_heads, dropout_rate
        )
        
        self.audio_encoder = TransformerAudioEncoder(
            audio_dim, hidden_dim // 2, num_layers // 2, num_heads // 2, dropout_rate
        )
        
        self.visual_encoder = TransformerVisualEncoder(
            visual_dim, hidden_dim // 2, num_layers // 2, num_heads // 2, dropout_rate
        )
        
        # Cross-attention fusion module
        self.fusion_module = MultimodalCrossAttention(
            hidden_dim, hidden_dim // 2, hidden_dim // 2, hidden_dim, num_heads, dropout_rate
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, text_features, audio_features, visual_features):
        # Encode each modality
        text_encoded = self.text_encoder.get_encoded_features(text_features)
        audio_encoded = self.audio_encoder.get_encoded_features(audio_features)
        visual_encoded = self.visual_encoder.get_encoded_features(visual_features)
        
        # Apply cross-attention fusion
        fused_features, sentiment = self.fusion_module(
            text_encoded, audio_encoded, visual_encoded
        )
        
        return sentiment