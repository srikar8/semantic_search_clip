import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), 
                         need_weights=False, attn_mask=attn_mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int = 224, patch_size: int = 32,
                 width: int = 512, layers: int = 12, heads: int = 8):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        
        self.transformer = nn.Sequential(*[
            ResidualAttentionBlock(width, heads) for _ in range(layers)
        ])
        
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor):
        # Shape: [batch_size, 3, resolution, resolution]
        x = self.conv1(x)  # Shape: [batch_size, width, grid, grid]
        
        # Reshape and permute
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Shape: [batch_size, width, grid ** 2]
        x = x.permute(0, 2, 1)  # Shape: [batch_size, grid ** 2, width]
        
        # Prepend class token
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # Shape: [batch_size, grid ** 2 + 1, width]
        
        # positional embeddings
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        # Transformer blocks
        x = x.permute(1, 0, 2)  # Shape: [grid ** 2 + 1, batch_size, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Shape: [batch_size, grid ** 2 + 1, width]
        
        # Return the transformed CLS token
        x = self.ln_post(x[:, 0, :])
        
        return x
    
class TextTransformer(nn.Module):
    def __init__(self, context_length: int = 77, vocab_size: int = 49408,
                 width: int = 512, layers: int = 12, heads: int = 8):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        
        self.transformer = nn.Sequential(*[
            ResidualAttentionBlock(width, heads) for _ in range(layers)
        ])
        
        self.ln_final = nn.LayerNorm(width)
        
        # Initialize the positional embeddings
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, text: torch.Tensor):
        # Shape: [batch_size, context_length]
        x = self.token_embedding(text)  # Shape: [batch_size, context_length, width]
        
        # positional embeddings
        x = x + self.positional_embedding
        
        # Transformer blocks
        x = x.permute(1, 0, 2)  # Shape: [context_length, batch_size, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Shape: [batch_size, context_length, width]
        
        # Layer norm and mean pooling
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        
        return x
    
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.vision_model = VisionTransformer()
        self.text_model = TextTransformer()

    def forward(self, image, text):
        # Process the image through the vision model
        image_features = self.encode_image(image)
        
        # Process the text through the text model
        text_features = self.encode_text(text)
        
        return image_features, text_features

    def encode_image(self, image):
        """
        Encodes an image using the VisionTransformer.
        
        Args:
            image: Tensor of shape (B, 3, H, W) representing the batch of images.
        
        Returns:
            A tensor of shape (B, N, embed_dim) representing the image embeddings.
        """
        image_features = self.vision_model(image)
        # Typically, we take the representation of the [CLS] token (first token) after passing through the transformer
        # But here we return the full sequence, which can also be processed later for image-text matching
        return image_features  # Return the [CLS] token or representation for image

    def encode_text(self, text):
        """
        Encodes text using the TextTransformer.
        
        Args:
            text: Tensor of shape (B, L) representing the tokenized input text.
        
        Returns:
            A tensor of shape (B, L, embed_dim) representing the text embeddings.
        """
        text_features = self.text_model(text)
        # Similar to images, we often take the representation of the [CLS] token
        return text_features  # Return the [CLS] token or representation for text