import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unfoldNd


class PatchEmbedding(nn.Module):
    def __init__(self, model_settings):
        """ 
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = (model_settings["patch_size"], model_settings["patch_size"], model_settings["patch_size"])
        self.num_channels = model_settings["num_channels"]
        self.embedding_size = model_settings["embedding_size"]
        
        self.num_patches = np.prod(model_settings["input_shape"]) // np.prod(self.patch_size) // self.num_channels
        
        self.linear1 = nn.Linear(np.prod(self.patch_size) * self.num_channels, self.embedding_size)

    def unfold(self, x):
        k, s = self.patch_size[0], self.patch_size[0]
        f= x.unfold(2, k, s).unfold(3, k, s).unfold(4, k, s)
        f= f[:, :, :, :, :, ::1, ::1, ::1]
        f = f.movedim(1,-1)
        f = torch.flatten(f, 1, 3)
        f = torch.flatten(f, 2, -1)
        return f
    
    def forward(self, x):
        #TODO: test with channels > 1
        x = self.unfold(x)          # Extract patches
        x = self.linear1(x)         # Create embeddings from patches
        
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, patch_embeddings, model_settings):
        super(PositionalEmbedding, self).__init__()
        self.num_embeddings = patch_embeddings
        self.embedding_dim = model_settings["embedding_size"]
        self.device = model_settings["device"]

        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim, device=self.device)

    def forward(self, x):
        # We add one extra embeddings, not used for position, but as the class embedding
        positional_ints = torch.arange(0, self.num_embeddings, requires_grad=False, device=self.device
                                       ).repeat(x.shape[0], 1)
        embedding = self.embedding(positional_ints)
        return embedding


class TransformerBlock(nn.Module):
    def __init__(self, model_settings, dropout=0.0):
        """ 
        """
        super(TransformerBlock, self).__init__()
        self.embedding_dim = model_settings["embedding_size"]
        self.heads = model_settings["attention_heads"]
        hidden_dim = self.embedding_dim * 2

        self.layer_norm_1 = nn.LayerNorm(self.embedding_dim)
        self.attention = nn.MultiheadAttention(self.embedding_dim, self.heads,
                                          dropout=dropout, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_dim)

        self.linear = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.embedding_dim),
            nn.Dropout(dropout))

    def forward(self, x_in):
        x = self.layer_norm_1(x_in)
        x = x_in + self.attention(x, x, x)[0]
        x = x + self.linear(self.layer_norm_2(x))

        return x   


class VIT(nn.Module):
    """ """

    def __init__(self, model_settings):
        """ 
        """
        super(VIT, self).__init__()
        self.num_channels = model_settings["num_channels"]
        self.input_shape = model_settings["input_shape"]
        self.embedding_size = model_settings["embedding_size"]
        self.device = model_settings["device"]
        self.patch_size = model_settings["patch_size"]

        self.input_conv = nn.Sequential(
            ConvolutionBlock(4, embedding_dim_out=8),
            ConvolutionBlock(8, embedding_dim_out=8)
        )

        self.patch_embedding = PatchEmbedding(model_settings)
        self.positional_embedding = PositionalEmbedding(self.patch_embedding.num_patches, model_settings)

        self.transformers = nn.ModuleList([TransformerBlock(model_settings) for i in range(model_settings["transformer_layers"])])

        self.decoder = ConvDecoder(model_settings)
        
        self.final_decoder = nn.Sequential(
            ConvolutionBlock(self.embedding_size//32 + 8),
            ConvolutionBlock(self.embedding_size//32 + 8),
            nn.Conv3d(self.embedding_size//32 + 8, 5, kernel_size=1)
        )
        



    def forward(self, x):
        x_input = x
        
        x = self.patch_embedding(x) 
        pos_embeddings = self.positional_embedding(x) # Also includes extra embeddings for class tokens
        
        x = x + pos_embeddings


        hidden_states = []
        save_at = [2,4,6]
        for i, block in enumerate(self.transformers):
            x = block(x)
            if i + 1 in save_at:
                c, i, j, k = self.input_shape
                hidden_states.append(x.reshape(x.shape[0], (i//self.patch_size), (j//self.patch_size), (k//self.patch_size), self.embedding_size).movedim(-1, 1))
    
        c, i, j, k = self.input_shape
        x = x.reshape(x.shape[0], (i//self.patch_size), (j//self.patch_size), (k//self.patch_size), self.embedding_size)
        x = x.movedim(-1, 1)
        x = self.decoder(x, hidden_states)
        convolutted_input = self.input_conv(x_input)
        x = torch.concat((x,convolutted_input), dim=1)
        x = self.final_decoder(x)
        
        return x
    

class ConvDecoder(nn.Module):
    def __init__(self, model_settings, dropout=0.1):
        """ 
        """
        super(ConvDecoder, self).__init__()
        self.embedding_dim = model_settings["embedding_size"]
        
        self.decoder0 = nn.ConvTranspose3d(self.embedding_dim, self.embedding_dim//2, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            ConvolutionBlock(self.embedding_dim, embedding_dim_out=self.embedding_dim//2),
            ConvolutionBlock(self.embedding_dim//2),
            nn.ConvTranspose3d(self.embedding_dim//2, self.embedding_dim//4, kernel_size=2, stride=2),
        )
        self.decoder2 = nn.Sequential(
            ConvolutionBlock(self.embedding_dim//2, embedding_dim_out=self.embedding_dim//4),
            ConvolutionBlock(self.embedding_dim//4),
            nn.ConvTranspose3d(self.embedding_dim//4, self.embedding_dim//8, kernel_size=2, stride=2),
        )
        self.decoder3 = nn.Sequential(
            ConvolutionBlock(self.embedding_dim//4, embedding_dim_out=self.embedding_dim//8),
            ConvolutionBlock(self.embedding_dim//8),
            nn.ConvTranspose3d(self.embedding_dim//8, self.embedding_dim//16, kernel_size=2, stride=2),
        )
        self.decoder4 = nn.Sequential(
            ConvolutionBlock(self.embedding_dim//16),
            ConvolutionBlock(self.embedding_dim//16),
            nn.ConvTranspose3d(self.embedding_dim//16, self.embedding_dim//32, kernel_size=2, stride=2),
        )

        self.feature_extractor_1 = HiddenExtractor(hidden_size=self.embedding_dim)
        self.feature_extractor_2 = nn.Sequential(
            HiddenExtractor(hidden_size=self.embedding_dim),
            HiddenExtractor(hidden_size=self.embedding_dim//2)
        )
        self.feature_extractor_3 = nn.Sequential(
            HiddenExtractor(hidden_size=self.embedding_dim),
            HiddenExtractor(hidden_size=self.embedding_dim//2),
            HiddenExtractor(hidden_size=self.embedding_dim//4)
        )
    

    def forward(self, x, hidden_states):
        x = self.decoder0(x)
        x = self.decoder1(torch.concat((x,self.feature_extractor_1(hidden_states[0])), dim=1))
        x = self.decoder2(torch.concat((x,self.feature_extractor_2(hidden_states[1])), dim=1))
        x = self.decoder3(torch.concat((x,self.feature_extractor_3(hidden_states[2])), dim=1))
        x = self.decoder4(x)
        return x


class ConvolutionBlock(nn.Module):
    def __init__(self, embedding_dim, embedding_dim_out=None):
        """ 
        """
        super(ConvolutionBlock, self).__init__()

        self.embedding_dim = embedding_dim
        if embedding_dim_out == None:
            self.embedding_dim_out = embedding_dim
        else:
            self.embedding_dim_out = embedding_dim_out

        self.layers = nn.Sequential(
            nn.Conv3d(self.embedding_dim, self.embedding_dim_out, kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.embedding_dim_out), # Test with batch size = 1...
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class HiddenExtractor(nn.Module):
    def __init__(self, hidden_size):
        super(HiddenExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose3d(hidden_size, hidden_size//2, kernel_size=2, stride=2),
            ConvolutionBlock(hidden_size//2)
        )

    def forward(self, x):
        return self.layers(x)