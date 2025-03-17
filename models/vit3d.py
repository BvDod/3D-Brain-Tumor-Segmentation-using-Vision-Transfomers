import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unfoldNd


class VIT3Dsegmentation(nn.Module):
    """ An implementation of the Vision Transformer (ViT) for 3D segmentation, based on the UNETR architecture. """

    def __init__(self, model_settings):
        super(VIT3Dsegmentation, self).__init__()
        self.num_channels = model_settings["num_channels"]
        self.input_shape = model_settings["input_shape"]
        self.embedding_size = model_settings["embedding_size"]
        self.device = model_settings["device"]
        self.patch_size = model_settings["patch_size"]
        self.output_classes = model_settings["output_classes"]

        #  Transformer layers at which we extract hidden states to pass to the decoder
        self.skip_at = [2, 4, 6]

        # Amount of dimensions of the features that are directly passed from input to the decoder
        # This calc maskes it the same size as the other decoder features at that stage
        self.input_skip_dim = self.embedding_size//32


        # Used to directly pass features from input to the last stage of the decoder
        self.input_conv = nn.Sequential(
            ConvolutionBlock(self.num_channels, embedding_dim_out=self.input_skip_dim),
            ConvolutionBlock(self.input_skip_dim)
        )

        self.patch_embedding = PatchEmbedding3D(model_settings)
        self.positional_embedding = PositionalEmbedding(self.patch_embedding.num_patches, model_settings)

        self.transformers = nn.ModuleList([TransformerBlock(model_settings) for i in range(model_settings["transformer_layers"])])

        self.decoder = ConvDecoder(model_settings)
        
        # Outputs the final segmentation mask
        self.final_decoder = nn.Sequential(
            ConvolutionBlock(self.embedding_size//32 + self.input_skip_dim),
            ConvolutionBlock(self.embedding_size//32 + self.input_skip_dim),
            nn.Conv3d(self.embedding_size//32 + self.input_skip_dim, self.output_classes, kernel_size=1)
        )
        

    def forward(self, x):
        x_input = x # Save for later

        x = self.patch_embedding(x) 
        pos_embeddings = self.positional_embedding(x) 
        x = x + pos_embeddings
        # Doesnt include class tokens, not needed for this task

        # Transfomer blocks: saves hidden states at the specified layers
        hidden_states = []
        for i, block in enumerate(self.transformers):
            x = block(x)
            if i + 1 in self.save_at:
                hidden_states.append(x.reshape(x.shape[0], # Reshape to 5D
                                    (self.input_shape[1]//self.patch_size), 
                                    (self.input_shape[2]//self.patch_size), 
                                    (self.input_shape[3]//self.patch_size), 
                                    self.embedding_size).movedim(-1, 1))
        # Reshape to 3D
        x = x.reshape(x.shape[0], 
                      (self.input_shape[1]//self.patch_size), 
                      (self.input_shape[2]//self.patch_size), 
                      (self.input_shape[3]//self.patch_size), 
                      self.embedding_size).movedim(-1, 1)

        x = self.decoder(x, hidden_states)
        
        convolutted_input = self.input_conv(x_input) # Extract features from input and concat to final decoder input
        x = torch.concat((x,convolutted_input), dim=1)
        
        x = self.final_decoder(x)
        return x
    

class PatchEmbedding3D(nn.Module):
    """ Implements the patch embedding layer. Creates embeddings from patches extracted from the input image. """
   
    def __init__(self, model_settings):
        super(PatchEmbedding3D, self).__init__()

        self.patch_size = (model_settings["patch_size"], model_settings["patch_size"], model_settings["patch_size"])
        self.num_channels = model_settings["num_channels"]
        self.embedding_size = model_settings["embedding_size"]
        
        # Nummber of patches that are going to be extracted
        self.num_patches = np.prod(model_settings["input_shape"]) // np.prod(self.patch_size) // self.num_channels
        
        # Creates embeddings from patches
        self.linear1 = nn.Linear(np.prod(self.patch_size) * self.num_channels, self.embedding_size)


    def unfold(self, x):
        k, s = self.patch_size[0], self.patch_size[0]

        # Basically we perform a 3D version of nn.Unfold here (only supports 4D tensors, not 5D)
        # Suprisingly, this works faster and with less memory than any 3d unfold implementation i could find
        # Source: https://github.com/pytorch/pytorch/issues/30798
        f= x.unfold(2, k, s).unfold(3, k, s).unfold(4, k, s)
        f= f[:, :, :, :, :, ::1, ::1, ::1]
        f = f.movedim(1,-1)
        f = torch.flatten(f, 1, 3)
        f = torch.flatten(f, 2, -1)

        return f
    
    def forward(self, x):
        x = self.unfold(x)          # Extract patches
        x = self.linear1(x)         # Create embeddings from patches  
        return x


class PositionalEmbedding(nn.Module):
    """ Implements the positional embedding layer. the positinal embeddings are learned"""

    def __init__(self, patch_embeddings, model_settings):
        super(PositionalEmbedding, self).__init__()

        self.num_embeddings = patch_embeddings
        self.embedding_dim = model_settings["embedding_size"]
        self.device = model_settings["device"]

        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim, device=self.device)

    def forward(self, x):
        # If this was a classification task, we would add a class token here, but we doint
        positional_ints = torch.arange(0, self.num_embeddings, requires_grad=False, device=self.device
                                       ).repeat(x.shape[0], 1)
        embedding = self.embedding(positional_ints)
        return embedding


class TransformerBlock(nn.Module):
    """ An implementation of the transformer block as described in the paper "Attention is All You Need"."""
    
    def __init__(self, model_settings, dropout=0.0):
        super(TransformerBlock, self).__init__()

        self.embedding_dim = model_settings["embedding_size"]
        self.heads = model_settings["attention_heads"]
        hidden_dim = self.embedding_dim * 2 # Hidden layer size in the feedforward network

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

    
class ConvDecoder(nn.Module):
    """ 
    The decoder part of the UNETR architecture, where the dimensions of image are gradually increased
    by deconvolution, while different hidden states from the encoder are passed at each stage.
    """

    def __init__(self, model_settings, dropout=0.1):

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
        # At this encoder, no hidden states are passed from encoder to decoder
        self.decoder4 = nn.Sequential(
            ConvolutionBlock(self.embedding_dim//16),
            ConvolutionBlock(self.embedding_dim//16),
            nn.ConvTranspose3d(self.embedding_dim//16, self.embedding_dim//32, kernel_size=2, stride=2),
        )

        # Extracts hidden states from the encoder
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
    """ Simple convolution block, !uses instance norm instead of batch norm! (UNETR uses batch norm)"""

    def __init__(self, embedding_dim, embedding_dim_out=None):
        super(ConvolutionBlock, self).__init__()

        self.embedding_dim = embedding_dim
        # If no output embedding size is specified, the output size is the same as the input size
        self.embedding_dim_out = embedding_dim if embedding_dim_out is None else embedding_dim_out
        
        # We use instance norm instead of batch norm, because we use batch size of 1 or 2 /:
        self.layers = nn.Sequential(
            nn.Conv3d(self.embedding_dim, self.embedding_dim_out, kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.embedding_dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class HiddenExtractor(nn.Module):
    """Used to extract hidden states from the encoder to pass to the decoder"""

    def __init__(self, hidden_size):
        super(HiddenExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose3d(hidden_size, hidden_size//2, kernel_size=2, stride=2),
            ConvolutionBlock(hidden_size//2)
        )

    def forward(self, x):
        return self.layers(x)