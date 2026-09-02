import torch
import math
import torch.nn as nn
from torch import Tensor
from typing import List, Callable, Optional
import numpy as np
from modelgenerator.adapters.base import (
    TokenAdapter,
    SequenceAdapter,
    ConditionalGenerationAdapter,
)
import torch.nn.functional as F
from peft import LoraConfig, LoraModel
from .swin_attn import window_partition, window_reverse, WindowAttention, RnaRMSNorm

class MLPAdapter(nn.Sequential, TokenAdapter):
    """Multi-layer perceptron (MLP) adapter.

    Args:
        in_features (int): Number of features of the input
        out_features (int): Number of features of the output
        hidden_sizes (List[int], optional): List of the hidden feature dimensions. Defaults to [].
        activation_layer (Callable[..., torch.nn.Module]): Activation function. Defaults to torch.nn.Tanh.
        bias (bool): Whether to use bias in the linear layer. Defaults to True
        dropout (float): The probability for the dropout layer. Defaults to 0.0
        dropout_in_middle (bool): Whether to use dropout in the middle layers. Defaults to True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: List[int] = [],
        activation_layer: Callable[..., torch.nn.Module] = torch.nn.Tanh,
        bias: bool = True,
        dropout: float = 0.0,
        dropout_in_middle: bool = True,
    ):
        layers = [nn.Dropout(dropout)]
        in_dim = in_features
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(activation_layer())
            if dropout_in_middle:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_features, bias=bias))

        super().__init__(*layers)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)

        Returns:
            torch.Tensor: predictions (n, seq_len, out_features)
        """
        output = super().forward(hidden_states)
        return output

class MLPAdapterWithoutOutConcat(nn.Module, TokenAdapter):
    """Multi-layer perceptron (MLP) adapter without outer concatenate
    
    This class is generally used in PairwiseTokenClassification. The following two implementations are equivalent:
    1. hidden_states -> outer_concat -> MLPAdapter
    2. hidden_states -> MLPAdapterWithoutOutConcat
    MLPAdapterWithoutOutConcat avoids the large memory consumption of outer_concat
    
    Args:
        in_features (int): Number of features of the input
        out_features (int): Number of features of the output
        hidden_sizes (List[int], optional): List of the hidden feature dimensions. Defaults to [].
        activation_layer (Callable[..., torch.nn.Module]): Activation function. Defaults to torch.nn.Tanh.
        bias (bool): Whether to use bias in the linear layer. Defaults to True
        dropout (float): The probability for the dropout layer. Defaults to 0.0
        dropout_in_middle (bool): Whether to use dropout in the middle layers. Defaults to True
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: List[int] = [],
        activation_layer: Callable[..., torch.nn.Module] = torch.nn.Tanh,
        bias: bool = True,
        dropout: float = 0.0,
        dropout_in_middle: bool = True,
    ):
        super().__init__()
        self.in_dropout = nn.Dropout(dropout)
        assert len(hidden_sizes) > 0, f"len(hidden_sizes) must be > 0"
        
        min_value = -np.sqrt(1 / (in_features*2))
        max_value =  np.sqrt(1 / (in_features*2))
        init_values = torch.FloatTensor(hidden_sizes[0], in_features*2).uniform_(min_value, max_value)
        self.first_linear = torch.nn.Parameter( init_values, requires_grad=True )
        init_values = torch.FloatTensor(hidden_sizes[0]).uniform_(min_value, max_value)
        self.first_bias   = torch.nn.Parameter( init_values, requires_grad=True )
        
        in_dim = hidden_sizes[0]
        
        layers = [ activation_layer() ]
        if dropout_in_middle:
            layers.append(nn.Dropout(dropout))
        for hidden_dim in hidden_sizes[1:]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(activation_layer())
            if dropout_in_middle:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, out_features, bias=bias))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)

        Returns:
            torch.Tensor: predictions (n, seq_len, seq_len, out_features)
        """
        n, seq_len, in_features = hidden_states.shape
        
        hidden_states = self.in_dropout(hidden_states) # [B, L, D]
        
        left  = hidden_states @ self.first_linear[:, 0::2].T # [B, L, E]
        right = hidden_states @ self.first_linear[:, 1::2].T # [B, L, E]
        
        hidden_states = left[..., :, None, :] + right[..., None, :, :] + self.first_bias[None, None, None, :] # [B, L, L, D1]
        
        I, J = torch.tril_indices(seq_len, seq_len, -1)
        hidden_states[..., I, J, :] = hidden_states[..., J, I, :]
        
        return self.layers(hidden_states)

class MLPPoolAdapter(nn.Module, SequenceAdapter):
    """MLP adapter for a 2D embedding with pooling for the sequence length dimension

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        pooling (str): Sequence pooling method. Defaults to mean_pooling
        hidden_sizes (List[int], optional): List of the hidden feature dimensions. Defaults to [].
        activation_layer (Callable[..., torch.nn.Module]): Activation function. Defaults to torch.nn.Tanh
        bias (bool): Whether to use bias in the linear layer. Defaults to True
        dropout (float): The probability for the dropout layer. Defaults to 0.0
        dropout_in_middle (bool): Whether to use dropout in the middle layers. Defaults to True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pooling: str = "mean_pooling",
        hidden_sizes: List[int] = [],
        activation_layer: Callable[..., torch.nn.Module] = torch.nn.Tanh,
        bias: bool = True,
        dropout: float = 0.0,
        dropout_in_middle: bool = True,
    ):
        super().__init__()
        self.pooling = pooling
        self.mlp = MLPAdapter(
            in_features,
            out_features,
            hidden_sizes,
            activation_layer,
            bias,
            dropout,
            dropout_in_middle,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        if self.pooling == "mean_pooling":
            if attention_mask is not None:
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states)
                )
                embeddings = torch.sum(
                    hidden_states * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            else:
                embeddings = hidden_states.mean(1)
        elif self.pooling == "cls_pooling":
            embeddings = hidden_states[:, 0]
        else:
            raise NotImplementedError
        output = self.mlp(embeddings)
        return output


class LinearAdapter(MLPAdapter):
    """Simple linear adapter for a 1D embedding

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, hidden_sizes=[])


class LinearCLSAdapter(nn.Module, SequenceAdapter):
    """Simple linear adapter for a 1D embedding

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        output = self.linear(hidden_states[:, 0])
        return output


class LinearMeanPoolAdapter(nn.Module, SequenceAdapter):
    """Mean pooling adapter for hidden_states of shape (n, seq_len, in_features)

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        if attention_mask is not None:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            embeddings = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        else:
            embeddings = torch.mean(hidden_states, dim=1)
        output = self.linear(embeddings)
        return output


class LinearMaxPoolAdapter(nn.Module, SequenceAdapter):
    """Simple max pooling adapter for [batch,seq_len,dim] embeddings

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            hidden_states_final = hidden_states.masked_fill(attention_mask == 0, -torch.inf)
        else:
            hidden_states_final = hidden_states
        embeddings = hidden_states_final.max(1)[0]
        return self.linear(embeddings)


class LinearTransformerAdapter(nn.Module, SequenceAdapter):
    """Transformer adapter

    Note: Support cls_pooling only.

    Args:
        embed_dim (int): Hidden size
        out_features (int): Number of output features
    """

    def __init__(self, embed_dim: int, out_features: int):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.linear = nn.Linear(embed_dim, out_features)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        x = self.transformer(hidden_states)
        embeddings = x[:, 0]  # cls pooling
        output = self.linear(embeddings)
        return output


class ConditionalLMAdapter(nn.Module, ConditionalGenerationAdapter):
    """Conditional sequence adapter

    Args:
        in_features (int): Number of input features
        embed_dim (int): Hidden size
        seq_len (int): Sequence length
    """

    def __init__(
        self,
        embed_dim: int,
        in_features: int,
        out_features: int,
        pretrained_decoder: nn.Module,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 2,
    ):
        super().__init__()
        self.label_encoder = nn.Linear(in_features, embed_dim)
        if pretrained_decoder is not None:
            self.lm_decoder = pretrained_decoder
        else:
            self.lm_decoder = nn.Linear(embed_dim, out_features)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers
        )

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Forward pass

        Args:
            embeddings (torch.Tensor): Input embeddings (n, seq_len, in_features)
            condition (torch.Tensor): Condition tensor (n, condition_features)

        Returns:
            torch.Tensor: Output embeddings (n, seq_len, out_features)
        """
        label_embeddings = self.label_encoder(labels)
        embeddings += label_embeddings.unsqueeze(1).tile(1, embeddings.shape[1], 1)
        transformed_embeddings_residual = self.transformer(embeddings)
        logits = self.lm_decoder(transformed_embeddings_residual + embeddings)
        return logits

class SDPAEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, self.nhead, 3 * self.d_head)
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, nhead, N, d_head]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.dropout(self.o_proj(attn_out))
        x = self.norm1(x)

        ff_out = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x

class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim, time_embed_dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_dim)

    def forward(self, x, t_emb):
        """
        x: (B, H, W, C) 输入特征
        t_emb: (B, time_embed_dim) 时间步嵌入
        """
        # LayerNorm
        h = self.norm(x)
        
        # 从时间嵌入生成 scale 和 shift
        scale, shift = self.linear(t_emb).chunk(2, dim=-1)  # (B, C), (B, C)

        scale = scale.unsqueeze(1).unsqueeze(1)
        shift = shift.unsqueeze(1).unsqueeze(1)

        return h * (1 + scale) + shift

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=64, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)
      
class ResNet2DAdapter(nn.Module, SequenceAdapter):
    """Adapter that applies ResNet2DModule to input embeddings.

    Args:
        in_channels (int): Input matrix channels.
        num_res_blocks (int): Number of residual blocks in the ResNet2DModule.
        conv_channels (int): Intermediate convolution channels.
        kernel_size (int): Kernel size for convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        num_res_blocks: int = 2,
        conv_channels: int = 64,
        kernel_size: int = 3,
        mode: str = 'latent_with_protenix_with_trm_deep_supervision_with_attn', # latent_with_protenix_with_trm # latent_with_protenix_with_trm_deep_supervision, latent_with_protenix_with_trm_deep_supervision_with_attn
        deep_supervision: bool = True,
        ab_triple: bool = False,
        ab_no_hierarchy: bool = False,
        residual: bool = False,
        lora: bool = False,
        unet: bool = False,
        tri_attn: bool = False,
        swin_transformer: bool = False,
        deep_supervision_ab_trm: bool = False
    ):
        super().__init__()

        # print('-----------------------------------------------')
        # print(f"mode: {mode}")
        # print(f"in_channels: {in_channels}, num_res_blocks: {num_res_blocks}") # 4096, 2
        # print('-----------------------------------------------')

        if not deep_supervision and not deep_supervision_ab_trm:
            self.in_to_conv = nn.Linear(in_channels, conv_channels)

        self.mode = mode
        self.lora = lora
        self.unet = unet
        self.tri_attn = tri_attn
        self.swin_transformer = swin_transformer
        self.deep_supervision = deep_supervision
        self.deep_supervision_ab_trm = deep_supervision_ab_trm
        self.residual = residual
        self.ab_triple = ab_triple
        self.ab_no_hierarchy = ab_no_hierarchy

        if self.mode == 'original':
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
        
        elif self.mode == "original_flash_attn":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

        elif self.mode == 'latent':
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

        elif self.mode == 'exp':
            self.resnet = self.ResNet2DModule(conv_channels + 1, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels + 1, 1, kernel_size=kernel_size, padding="same"
            )
            self.no_cycling_linear = nn.Linear(conv_channels + 1, conv_channels + 1, bias=False)

        elif self.mode == "latent_with_exp":
            self.resnet = self.ResNet2DModule(conv_channels + 1, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels + 1, 1, kernel_size=kernel_size, padding="same"
            )
            self.no_cycling_linear = nn.Linear(conv_channels + 1, conv_channels + 1, bias=False)

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_emb_linear = nn.Linear(conv_channels, conv_channels, bias=False)
        
        elif self.mode == "v1":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)
            self.no_cycling_pred_linear = nn.Linear(1, conv_channels, bias=False)

        elif self.mode == 'latent_with_attn':
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=conv_channels, 
                nhead=8, 
                dim_feedforward=int(conv_channels * 4), 
                batch_first=True,   # [B, N, C]
                norm_first=False
            )
            
            self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        elif self.mode == "latent_with_exp_attn":
            self.resnet = self.ResNet2DModule(conv_channels + 1, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels + 1, 1, kernel_size=kernel_size, padding="same"
            )
            self.no_cycling_linear = nn.Linear(conv_channels + 1, conv_channels + 1, bias=False)

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_emb_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=conv_channels + 1, 
                nhead=8, 
                dim_feedforward=int((conv_channels + 1) * 4), 
                batch_first=True,   # [B, N, C]
                norm_first=False
            )
            
            self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        elif mode == "latent_with_flash_attn":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

        elif mode == "latent_with_flash_attn_with_posemb":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

            self.positional_embeddings = nn.ModuleList([
                nn.Embedding(1, conv_channels) for _ in range(5)
            ])
        
        elif mode == "latent_with_flash_attn_with_posemb_ahead":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

            self.positional_embeddings = nn.ModuleList([
                nn.Embedding(1, conv_channels) for _ in range(5)
            ])

        elif mode == "latent_with_flash_attn_with_posemb_with_residue":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

            self.positional_embeddings = nn.ModuleList([
                nn.Embedding(1, conv_channels) for _ in range(5)
            ])

        elif mode == "latent_with_flash_attn_with_tokenrouter":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

            # self.positional_embeddings = nn.ModuleList([
            #     nn.Embedding(1, conv_channels) for _ in range(5)
            # ])

            self.token_routers = nn.ModuleList([
                nn.Linear(conv_channels, 1, bias=True) for _ in range(5)
            ])

        elif mode == "latent_with_flash_attn_with_adalayernorm":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            
            self.recycle_embed = nn.Embedding(5, conv_channels)
            self.no_cycling_norm = AdaLayerNorm(hidden_dim = conv_channels, time_embed_dim = conv_channels)

            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

        elif mode == "latent_with_flash_attn_with_adalayernorm_with_tokenrouter":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            
            self.recycle_embed = nn.Embedding(5, conv_channels)
            self.no_cycling_norm = AdaLayerNorm(hidden_dim = conv_channels, time_embed_dim = conv_channels)

            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

            self.token_routers = nn.ModuleList([
                nn.Linear(conv_channels, 1, bias=True) for _ in range(5)
            ])

        elif self.mode == "latent_with_flash_attn_with_5layernorm":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

            # self.positional_embeddings = nn.ModuleList([
            #     nn.Embedding(1, conv_channels) for _ in range(5)
            # ])

            self.no_cycling_norm = nn.ModuleList([
                nn.LayerNorm(conv_channels) for _ in range(5)
            ])

        elif self.mode == "latent_with_flash_attn_with_dit_layernorm":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            
            self.recycle_embed = nn.Embedding(5, conv_channels)

            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

            self.norm_resnet = nn.LayerNorm(conv_channels, elementwise_affine=False)
            self.norm_attn = nn.LayerNorm(conv_channels, elementwise_affine=False)

            self.linear = nn.Linear(conv_channels, 6 * conv_channels)

        # elif self.mode == "latent_with_protenix":
        #     if lora:
        #         self.block = self.Block(conv_channels, num_res_blocks, kernel_size)
        #         # lora_config = LoraConfig(target_modules=["conv", "linear"], r=32, lora_alpha=32, lora_dropout=0.0)
        #         lora_config = LoraConfig(target_modules=["conv", "linear"], lora_dropout=0.0)

        #         self.loras = nn.ModuleList([
        #             LoraModel(self.block, lora_config, adapter_name=f"lora_{i}")
        #             for i in range(5)
        #         ])

        #         self.conv_to_output = nn.Conv2d(
        #             conv_channels, 1, kernel_size=kernel_size, padding="same"
        #         )
        #     else:
        #         if unet:
        #             self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        #             self.conv1 = self.conv_block(ch_in=64,ch_out=128)
        #             self.conv2 = self.conv_block(ch_in=128,ch_out=256)
        #             self.conv3 = self.conv_block(ch_in=256,ch_out=512)
        #             self.conv4 = self.conv_block(ch_in=512,ch_out=1024)
        #             # self.conv5 = self.conv_block(ch_in=1024,ch_out=2048)

        #             # self.up5 = self.up_conv(ch_in=2048,ch_out=1024)
        #             # self.up_conv5 = self.conv_block(ch_in=2048,ch_out=1024)

        #             self.up4 = self.up_conv(ch_in=1024,ch_out=512)
        #             self.up_conv4 = self.conv_block(ch_in=1024,ch_out=512)
                    
        #             self.up3 = self.up_conv(ch_in=512,ch_out=256)
        #             self.up_conv3 = self.conv_block(ch_in=512,ch_out=256)
                    
        #             self.up2 = self.up_conv(ch_in=256,ch_out=128)
        #             self.up_conv2 = self.conv_block(ch_in=256,ch_out=128)

        #             self.up1 = self.up_conv(ch_in=128,ch_out=64)
        #             self.up_conv1 = self.conv_block(ch_in=128,ch_out=64)

        #             self.conv_1x1 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)

        #             self.no_cycling_norm = nn.LayerNorm(conv_channels)
        #             self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)
                    
        #         elif tri_attn:
        #             self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
        #             self.no_cycling_norm = nn.LayerNorm(conv_channels)
        #             self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

        #             self.conv_to_output = nn.Conv2d(
        #                 conv_channels, 1, kernel_size=kernel_size, padding="same"
        #             )

        #             self.tri_out = self.TriOut(64)
        #             self.tri_in = self.TriIn(64)
        #             # self.tri_attn_start1 = self.TriAttStart(64)
        #             # self.tri_attn_end1 = self.TriAttEnd(64)

        #             # self.tri_attn_start2 = self.TriAttStart(64)
        #             # self.tri_attn_end2 = self.TriAttEnd(64)

        #         elif swin_transformer:
        #             self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
        #             self.no_cycling_norm = nn.LayerNorm(conv_channels)
        #             self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

        #             self.conv_to_output = nn.Conv2d(
        #                 conv_channels, 1, kernel_size=kernel_size, padding="same"
        #             )

        #             self.swin_transformer = self.SwinTransformer()

        #         else:
        #             self.x_init_linear = nn.Linear(conv_channels, conv_channels)

        #             self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
        #             self.no_cycling_norm = nn.LayerNorm(conv_channels)
        #             self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

        #             self.conv_to_output = nn.Conv2d(
        #                 conv_channels, 1, kernel_size=kernel_size, padding="same"
        #             )

        #             # self.attn_encoder = nn.Sequential(
        #             #     *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
        #             # )

        #             # nn.init.zeros_(self.no_cycling_linear.weight)
        
        elif self.mode == "latent_with_protenix_with_dit_layernorm":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            
            self.recycle_embed = nn.Embedding(5, conv_channels)

            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

            self.norm_resnet = nn.LayerNorm(conv_channels, elementwise_affine=False)
            self.norm_attn = nn.LayerNorm(conv_channels, elementwise_affine=False)
            self.norm_recycle = nn.LayerNorm(conv_channels, elementwise_affine=False)

            self.linear = nn.Sequential(
                nn.SiLU(),
                nn.Linear(conv_channels, 6 * conv_channels, bias=True)
            )

            nn.init.zeros_(self.no_cycling_linear.weight)

        elif self.mode == "latent_with_protenix_with_dit_layernorm_trm":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            
            self.recycle_embed = nn.Embedding(6, conv_channels)

            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

            self.norm_resnet = nn.LayerNorm(conv_channels, elementwise_affine=False)
            self.norm_attn = nn.LayerNorm(conv_channels, elementwise_affine=False)
            self.norm_recycle = nn.LayerNorm(conv_channels, elementwise_affine=False)

            self.linear = nn.Sequential(
                nn.SiLU(),
                nn.Linear(conv_channels, 4 * conv_channels, bias=True)
            )

            self.recycle_embed_k = nn.Embedding(3, conv_channels)

            nn.init.zeros_(self.no_cycling_linear.weight)

        elif self.mode == "latent_with_protenix_with_trm":
            # self.resnet_x = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)

            # self.no_cycling_linear_x = nn.Linear(conv_channels, conv_channels, bias=False)
            # # self.attn_encoder = nn.Sequential(
            # #     *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            # # )
            # self.norm_recycle_x = nn.LayerNorm(conv_channels)
            
            # self.resnet_y = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)

            # self.no_cycling_linear_y = nn.Linear(conv_channels, conv_channels, bias=False)

            # self.norm_recycle_y = nn.LayerNorm(conv_channels)

            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)

            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.norm_recycle = nn.LayerNorm(conv_channels)
        
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.linear_x = nn.Linear(conv_channels, conv_channels)
            self.linear_y = nn.Linear(conv_channels, conv_channels)
        
        
        elif self.mode == "latent_with_protenix_with_trm_3loss":

            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)

            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.norm_recycle = nn.LayerNorm(conv_channels)
        
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.linear_x = nn.Linear(conv_channels, conv_channels)
            self.linear_y = nn.Linear(conv_channels, conv_channels)

        elif self.mode == "latent_with_protenix_with_ab_trm_deep_supervision":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            # self.norm_recycle = nn.LayerNorm(conv_channels)

        elif self.mode == "latent_with_protenix_with_trm_deep_supervision":
            #
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)
            # self.attn_encoder = nn.Sequential(
            #     *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            # )
            self.norm_recycle = nn.LayerNorm(conv_channels)
            # nn.init.zeros_(self.no_cycling_linear.weight)

        elif self.mode == "latent_with_protenix_with_trm_deep_supervision_with_attn":
            if self.ab_triple or self.ab_no_hierarchy:
                self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            else:
                self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size, use_sequence=True)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)
            self.norm_recycle = nn.LayerNorm(conv_channels)
            # self.protenix_residual = MLP(in_dim=128, hidden_dim=256, out_dim=1)

        elif self.mode == "latent_with_protenix_with_deep_supervision":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)
            # self.attn_encoder = nn.Sequential(
            #     *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            # )
            self.no_cycling_norm = nn.LayerNorm(conv_channels)

        elif self.mode == "latent_with_protenix_tri_flash_attn":
            self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
            self.no_cycling_norm = nn.LayerNorm(conv_channels)
            self.no_cycling_linear = nn.Linear(conv_channels, conv_channels, bias=False)

            self.conv_to_output = nn.Conv2d(
                conv_channels, 1, kernel_size=kernel_size, padding="same"
            )

            self.attn_encoder = nn.Sequential(
                *[SDPAEncoderLayer(conv_channels, 2, conv_channels * 4) for _ in range(2)]
            )

    def symmetrize_matrix(self, matrix: Tensor) -> Tensor:
        """
        Symmetrizes a square matrix by preserving the upper triangular part
        and mirroring it across the diagonal.

        Args:
            matrix (Tensor): Input tensor of shape (B, H, W), where H = W

        Returns:
            Tensor: Symmetrized tensor of shape (B, H, W).
        """
        upper_triangular = torch.triu(matrix, diagonal=1)
        return upper_triangular + upper_triangular.transpose(-1, -2)

    def _generate_sharp_loop_mask(self, seq_len):
        mask = np.eye(seq_len, k=0, dtype=bool)
        for i in range(1, 4):
            mask = (
                mask + np.eye(seq_len, k=i, dtype=bool) + np.eye(seq_len, k=-i, dtype=bool)
            )
        return mask

    def _generate_canonical_pairs_mask(self, seq: str):
        seq = seq.replace("T", "U")

        mask = np.zeros((len(seq), len(seq)), dtype=bool)
        for i, nt_i in enumerate(seq):
            for j, nt_j in enumerate(seq):
                if f"{nt_i}{nt_j}" in ["AU", "UA", "GC", "CG", "GU", "UG"]:
                    mask[i, j] = True

        return mask

    def latent_recursion(self, x_init, x, y, no_cycling, k):
        t_emb_k = self.recycle_embed_k(k)
        t_emb_list = [torch.tensor([i], dtype=torch.long, device=x.device) for i in range(no_cycling + 1)]

        for i in range(no_cycling):
            t_emb = self.recycle_embed(t_emb_list[i])
            attn_scale, attn_shift, \
            recycle_scale, recycle_shift = self.linear(t_emb + t_emb_k).chunk(4, dim=-1)  # (B, C), (B, C)

            attn_scale = attn_scale.unsqueeze(1).unsqueeze(1)
            attn_shift = attn_shift.unsqueeze(1).unsqueeze(1)
            recycle_scale = recycle_scale.unsqueeze(1).unsqueeze(1)
            recycle_shift = recycle_shift.unsqueeze(1).unsqueeze(1)

            x = x_init + self.no_cycling_linear(self.norm_recycle(x + y) * (1 + recycle_scale) + recycle_shift)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.resnet(x)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

            B, H, W, C = x.shape
            x = self.norm_attn(x) * (1 + attn_scale) + attn_shift
            x = x.view(B, H * W, C)
            x = self.attn_encoder(x)
            x = x.view(B, H, W, C)

        t_emb = self.recycle_embed(t_emb_list[no_cycling])
        attn_scale_k, attn_shift_k, \
        recycle_scale_k, recycle_shift_k = self.linear(t_emb + t_emb_k).chunk(4, dim=-1)  # (B, C), (B, C)
        attn_scale_k = attn_scale_k.unsqueeze(1).unsqueeze(1)
        attn_shift_k = attn_shift_k.unsqueeze(1).unsqueeze(1)
        recycle_scale_k = recycle_scale_k.unsqueeze(1).unsqueeze(1)
        recycle_shift_k = recycle_shift_k.unsqueeze(1).unsqueeze(1)

        y = self.no_cycling_linear(self.norm_recycle(x + y) * (1 + recycle_scale_k) + recycle_shift_k)

        y = y.permute(0, 3, 1, 2)  # (B, C, H, W)
        y = self.resnet(y)
        y = y.permute(0, 2, 3, 1)  # (B, H, W, C)

        B, H, W, C = y.shape
        y = self.norm_attn(y) * (1 + attn_scale_k) + attn_shift_k

        y = y.view(B, H * W, C)
        y = self.attn_encoder(y)
        y = y.view(B, H, W, C)
        return x, y
    
    def latent_recursion_wo_time_emb(self, x_init, x, y, z, no_cycling, sequence_x=None, sequence_y=None, z_no_cycling=None):
        if self.ab_triple:
            for _ in range(z_no_cycling + 1):
                for _ in range(no_cycling):
                    x = self.no_cycling_linear(self.norm_recycle(x + y + z + x_init))
                    x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                    x = self.resnet(x)
                    x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
                z = self.no_cycling_linear(self.norm_recycle(x + y + z))
                z = z.permute(0, 3, 1, 2)  # (B, C, H, W)
                z = self.resnet(z)
                z = z.permute(0, 2, 3, 1)  # (B, H, W, C)
            return x, y, z
        
        else:
            # # 1. 一套resnet
            for _ in range(no_cycling):
                x = self.no_cycling_linear(self.norm_recycle(x + y + x_init))
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x, sequence=sequence_x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

                # B, H, W, C = x.shape
                # x = x.view(B, H * W, C)
                # x = self.attn_encoder(x)
                # x = x.view(B, H, W, C)

            y = self.no_cycling_linear(self.norm_recycle(x + y))

            y = y.permute(0, 3, 1, 2)  # (B, C, H, W)
            y = self.resnet(y, sequence=sequence_y)
            y = y.permute(0, 2, 3, 1)  # (B, H, W, C)

            # B, H, W, C = y.shape
            # y = y.view(B, H * W, C)
            # y = self.attn_encoder(y)
            # y = y.view(B, H, W, C)

            # 2. 两套resnet, 不收敛
            # for _ in range(no_cycling):
            #     x = self.no_cycling_linear_x(self.norm_recycle_x(x + y + x_init))
            #     x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            #     x = self.resnet_x(x)
            #     x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

            #     # B, H, W, C = x.shape
            #     # x = x.view(B, H * W, C)
            #     # x = self.attn_encoder(x)
            #     # x = x.view(B, H, W, C)

            # y = self.no_cycling_linear_y(self.norm_recycle_y(x + y))

            # y = y.permute(0, 3, 1, 2)  # (B, C, H, W)
            # y = self.resnet_y(y)
            # y = y.permute(0, 2, 3, 1)  # (B, H, W, C)

            # # B, H, W, C = y.shape
            # # y = y.view(B, H * W, C)
            # # y = self.attn_encoder(y)
            # # y = y.view(B, H, W, C)

            return x, y, z

    def rms_norm(self, hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        return hidden_states.to(input_dtype)
    
    def latent_recursion_ab_trm_deep_supervision(self, x_init, x, y, no_cycling):
        for _ in range(no_cycling):
            x = x + y + x_init
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.resnet(x)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            # x = self.norm_recycle(x)
            x = self.rms_norm(x, variance_epsilon=1e-5)
            
        y = x + y
        y = y.permute(0, 3, 1, 2)  # (B, C, H, W)
        y = self.resnet(y)
        y = y.permute(0, 2, 3, 1)  # (B, H, W, C)
        # y = self.norm_recycle(y)
        y = self.rms_norm(y, variance_epsilon=1e-5)

        return x, y
    
    def latent_recursion_wo_time_emb_no_xinit(self, x, y, no_cycling):
        for _ in range(no_cycling):
            x = self.no_cycling_linear(self.norm_recycle(x + y))
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.resnet(x)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

            # B, H, W, C = x.shape
            # x = x.view(B, H * W, C)
            # x = self.attn_encoder(x)
            # x = x.view(B, H, W, C)

        y = self.no_cycling_linear(self.norm_recycle(x + y))

        y = y.permute(0, 3, 1, 2)  # (B, C, H, W)
        y = self.resnet(y)
        y = y.permute(0, 2, 3, 1)  # (B, H, W, C)

        # B, H, W, C = y.shape
        # y = y.view(B, H * W, C)
        # y = self.attn_encoder(y)
        # y = y.view(B, H, W, C)

        return x, y

    def row_col_softmax(self, y):
        row_softmax = torch.softmax(y, dim=-1)
        col_softmax = torch.softmax(y, dim=-2)
        return 0.5 * (row_softmax + col_softmax)
    
    def forward(self, x: Tensor, attention_mask: Tensor = None, no_cycling: int = 5, k_no_cycling: int = 2, z_no_cycling: int = 2, seq: str = None, x_recycle=None, y_recycle=None, z_recycle=None, sequence_x=None, sequence_y=None, protenix_embedding=None, residual_weight=0.0, cls_token=None) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): Input tensor of shape (B, H, W, E).

        Returns:
            Tensor: Symmetrized output tensor of shape (B, H, W).
        """
        if not self.deep_supervision and not self.deep_supervision_ab_trm:
            x = self.in_to_conv(x) # [batch_size, seq_len, seq_len, features * 2] -> [batch_size, seq_len, seq_len, 64]
        
        if self.mode == "original":
            x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
            x = self.resnet(x)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # Reduce to (B, H, W).
            x = self.symmetrize_matrix(x)

            # if with rfold loss
            # x = self.row_col_softmax(x)
            return x
        
        elif self.mode == "original_flash_attn":
            x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
            x = self.resnet(x)

            x = x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).
            B, H, W, C = x.shape
            x = x.view(B, H*W, C)
            x = self.attn_encoder(x)
            x = x.view(B, H, W, C)
            x = x.permute(0, 3, 1, 2)

            x = self.conv_to_output(x)
            x = x.squeeze(1)  # Reduce to (B, H, W).
            x = self.symmetrize_matrix(x)
            return x   
          
        elif self.mode == "latent":
            for _ in range(no_cycling):
                x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).

                x = x + self.no_cycling_linear(self.no_cycling_norm(x))
                
            x = x.permute(0, 3, 1, 2)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # Reduce to (B, H, W).
            x = self.symmetrize_matrix(x)
            
            return x
        
        elif self.mode == "exp":
            # breakpoint()
            x_shape = x.shape # B, H, W, 64
            initial_no_cycling = torch.zeros(*x_shape[:-1], 1, device=x.device) # Convert to (B, H, W, 1)
            
            for _ in range(no_cycling):
                cycle_x = torch.cat([initial_no_cycling, x], dim=-1) # 1, h, w, 65
                cycle_x = self.no_cycling_linear(cycle_x) # # 1, h, w, 65

                cycle_x = cycle_x.permute(0, 3, 1, 2) # 1, 65, h, w
                cycle_x = self.resnet(cycle_x) # 1, 65, h, w
                
                cycle_x = self.conv_to_output(cycle_x) # 1, 1, h, w
                cycle_x = cycle_x.squeeze(1)  # Reduce to (B, H, W).
                cycle_x = self.symmetrize_matrix(cycle_x) # 1, h, w

                # constraint
                # allowed_pairs_mask = np.logical_not(np.eye(len(seq), dtype=bool))
                # allowed_pairs_mask = np.logical_and(
                #     allowed_pairs_mask, ~self._generate_sharp_loop_mask(len(seq))
                # )
                # allowed_pairs_mask = np.logical_and(
                #     allowed_pairs_mask, self._generate_canonical_pairs_mask(seq)
                # )

                # allowed_pairs_mask = np.expand_dims(allowed_pairs_mask, axis=0)

                # allowed_pairs_mask = torch.tensor(allowed_pairs_mask, dtype=torch.bool, device=x.device)
                # cycle_x = cycle_x * allowed_pairs_mask

                # cycle_x = self.symmetrize_matrix(cycle_x) # (B, H, W)

                return_res = cycle_x.clone() # (1, h, w)

                cycle_x = torch.sigmoid(cycle_x)
                initial_no_cycling = cycle_x.unsqueeze(-1) # (B, H, W, 1)

            # breakpoint()
            print("Logits min/max:", return_res.min().item(), return_res.max().item())
            return return_res
        
        elif self.mode == "latent_with_exp":
            x_shape = x.shape # B, H, W, 64
            # initial_no_cycling = torch.zeros(*x_shape[:-1], 1, device=x.device) # Convert to (B, H, W, 1)
            initial_no_cycling = torch.full((*x_shape[:-1], 1), -1, device=x.device)
            
            for _ in range(no_cycling):
                cycle_x = torch.cat([initial_no_cycling, x], dim=-1) # 1, h, w, 65
                cycle_x = self.no_cycling_linear(cycle_x) # 1, h, w, 65

                cycle_x = cycle_x.permute(0, 3, 1, 2) # 1, 65, h, w
                cycle_x = self.resnet(cycle_x) # 1, 65, h, w

                x = x + self.no_cycling_emb_linear(self.no_cycling_norm(cycle_x.permute(0, 2, 3, 1)[:, :, :, 1:]))

                cycle_x = self.conv_to_output(cycle_x) # 1, 1, h, w
                cycle_x = cycle_x.squeeze(1)  # Reduce to (B, H, W).
                cycle_x = self.symmetrize_matrix(cycle_x)

                return_res = cycle_x.clone()

                cycle_x = torch.sigmoid(cycle_x)
                initial_no_cycling = cycle_x.unsqueeze(-1) # (B, H, W, 1)

            return return_res
        
        elif self.mode == "v1":
            for _ in range(no_cycling):
                x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
                x = self.resnet(x) # 1, 64, n, n

                logits = self.conv_to_output(x)
                logits = logits.squeeze(1)  # Reduce to (B, H, W).
                logits = self.symmetrize_matrix(logits)

                # allowed_pairs_mask = np.logical_not(np.eye(len(seq), dtype=bool))
                # allowed_pairs_mask = np.logical_and(
                #     allowed_pairs_mask, ~self._generate_sharp_loop_mask(len(seq))
                # )
                # allowed_pairs_mask = np.logical_and(
                #     allowed_pairs_mask, self._generate_canonical_pairs_mask(seq)
                # )
                # allowed_pairs_mask = np.expand_dims(allowed_pairs_mask, axis=0)
                # allowed_pairs_mask = torch.tensor(allowed_pairs_mask, dtype=torch.bool, device=x.device)
                # logits = logits * allowed_pairs_mask
                # logits = self.symmetrize_matrix(logits)

                preds = torch.sigmoid(logits) # b h w
                preds = preds.unsqueeze(-1)

                x = x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).
                x = x + self.no_cycling_linear(self.no_cycling_norm(x)) # b h w c
                x = x + self.no_cycling_pred_linear(preds)

            return logits

        elif self.mode == "latent_with_attn":
            for _ in range(no_cycling):
                x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).

                B, H, W, C = x.shape
                x = x.view(B, H*W, C)

                x = self.attn_encoder(x)

                x = x.view(B, H, W, C)
                x = x + self.no_cycling_linear(self.no_cycling_norm(x))
                
            x = x.permute(0, 3, 1, 2)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # Reduce to (B, H, W).
            x = self.symmetrize_matrix(x)
            return x
        
        elif self.mode == "latent_with_exp_attn":
            x_shape = x.shape # B, H, W, 64
            initial_no_cycling = torch.zeros(*x_shape[:-1], 1, device=x.device) # logits (B, H, W, 1)

            for _ in range(no_cycling):
                cycle_x = torch.cat([initial_no_cycling, x], dim=-1) # 1, h, w, 65
                cycle_x = self.no_cycling_linear(cycle_x) # 1, h, w, 65

                cycle_x = cycle_x.permute(0, 3, 1, 2)  # Convert to (B, 65, H, W).
                cycle_x = self.resnet(cycle_x)
                cycle_x = cycle_x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).

                B, H, W, C = cycle_x.shape
                cycle_x = cycle_x.view(B, H*W, C)

                cycle_x = self.attn_encoder(cycle_x)

                cycle_x = cycle_x.view(B, C, H, W) # 1 65 H W
                x = x + self.no_cycling_emb_linear(self.no_cycling_norm(cycle_x.permute(0, 2, 3, 1)[:, :, :, 1:]))
                
                cycle_x = self.conv_to_output(cycle_x)
                cycle_x = cycle_x.squeeze(1)  # Reduce to (B, H, W).
                cycle_x = self.symmetrize_matrix(cycle_x)

                return_res = cycle_x.clone()
                cycle_x = torch.sigmoid(cycle_x)
                initial_no_cycling = cycle_x.unsqueeze(-1) # (B, H, W, 1)

            return return_res

        elif self.mode == "latent_with_flash_attn":
            for _ in range(no_cycling):
                x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).

                B, H, W, C = x.shape
                x = x.view(B, H*W, C)

                x = self.attn_encoder(x)

                x = x.view(B, H, W, C)
                x = x + self.no_cycling_linear(self.no_cycling_norm(x))
                
            x = x.permute(0, 3, 1, 2)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # Reduce to (B, H, W).
            x = self.symmetrize_matrix(x)
            
            return x
        
        elif self.mode == "latent_with_flash_attn_with_posemb":
            for i in range(no_cycling):
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

                B, H, W, C = x.shape

                # broadcast positional embedding: (C,) -> (1,1,1,C) -> (B,H,W,C)
                pos_emb = self.positional_embeddings[i](torch.zeros(1, dtype=torch.long, device=x.device))
                x = x * pos_emb.view(1, 1, 1, C) # 放在self.no_cycling_linear(self.no_cycling_norm(x))前面还是后面

                x = x.view(B, H * W, C)
                x = self.attn_encoder(x)
                x = x.view(B, H, W, C)

                x = x + self.no_cycling_linear(self.no_cycling_norm(x))

                # pos_emb = self.positional_embeddings[i](torch.zeros(1, dtype=torch.long, device=x.device))
                # x = x * pos_emb.view(1, 1, 1, C)

            x = x.permute(0, 3, 1, 2)  # (B,C,H,W)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # (B,H,W)
            x = self.symmetrize_matrix(x)
            return x

        elif self.mode == "latent_with_flash_attn_with_posemb_ahead":
            for i in range(no_cycling):
                pos_emb = self.positional_embeddings[i](torch.zeros(1, dtype=torch.long, device=x.device))
                x = x * pos_emb.view(1, 1, 1, C)

                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

                B, H, W, C = x.shape

                x = x.view(B, H * W, C)
                x = self.attn_encoder(x)
                x = x.view(B, H, W, C)

                x = x + self.no_cycling_linear(self.no_cycling_norm(x))

            x = x.permute(0, 3, 1, 2)  # (B,C,H,W)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # (B,H,W)
            x = self.symmetrize_matrix(x)
            return x
        
        elif self.mode == "latent_with_flash_attn_with_posemb_with_residue":
            residue = torch.zeros_like(x, device=x.device)
            for i in range(no_cycling):
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

                B, H, W, C = x.shape

                # broadcast positional embedding: (C,) -> (1,1,1,C) -> (B,H,W,C)
                pos_emb = self.positional_embeddings[i](torch.zeros(1, dtype=torch.long, device=x.device))
                x = x * pos_emb.view(1, 1, 1, C) # 放在self.no_cycling_linear(self.no_cycling_norm(x))前面还是后面

                x = x.view(B, H * W, C)
                x = self.attn_encoder(x)
                x = x.view(B, H, W, C)

                x = x + self.no_cycling_linear(self.no_cycling_norm(x))

                residue = residue + x
                x = residue

            x = x.permute(0, 3, 1, 2)  # (B,C,H,W)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # (B,H,W)
            x = self.symmetrize_matrix(x)
            return x
        
        elif self.mode == "latent_with_flash_attn_with_tokenrouter":
            last_x = None

            for i in range(no_cycling):
                # pos_emb = self.positional_embeddings[i](torch.zeros(1, dtype=torch.long, device=x.device))
                # x = x * pos_emb.view(1, 1, 1, C)
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

                B, H, W, C = x.shape

                x = x.view(B, H * W, C)
                x = self.attn_encoder(x)
                x = x.view(B, H, W, C)

                x = x + self.no_cycling_linear(self.no_cycling_norm(x))

                # w = torch.sigmoid(self.token_routers[i](x))

                # soft routing
                # if last_x is not None:
                #     x = w * x + (1 - w) * last_x

                # hard routing
                if last_x is not None:
                    logits = self.token_routers[i](x) # b h w 1
                    U = torch.rand_like(logits)
                    gumbel_noise = -torch.log(-torch.log(U + 1e-9) + 1e-9)
                    w = torch.sigmoid((logits + gumbel_noise) / 0.5)
                    x = w * x + (1 - w) * last_x

                    # mask = (w < 0.5).unsqueeze(-1)  # (B,H,W,1)
                    # x = torch.where(mask, last_x, x)

                last_x = x.clone()

            x = x.permute(0, 3, 1, 2)  # (B,C,H,W)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # (B,H,W)
            x = self.symmetrize_matrix(x)
            return x
        
        elif self.mode == "latent_with_flash_attn_with_adalayernorm_with_tokenrouter":
            last_x = None
            for i in range(no_cycling):
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

                B, H, W, C = x.shape

                x = x.view(B, H * W, C)
                x = self.attn_encoder(x)
                x = x.view(B, H, W, C)

                t_emb = self.recycle_embed(torch.tensor([i], dtype=torch.long, device=x.device))
                x = x + self.no_cycling_linear(self.no_cycling_norm(x, t_emb))

                if last_x is not None:
                    logits = self.token_routers[i](x) # b h w 1
                    U = torch.rand_like(logits)
                    gumbel_noise = -torch.log(-torch.log(U + 1e-9) + 1e-9)
                    w = torch.sigmoid((logits + gumbel_noise) / 0.5)
                    x = w * x + (1 - w) * last_x
                    # mask = (w < 0.5).unsqueeze(-1)  # (B,H,W,1)
                    # x = torch.where(mask, last_x, x)
                last_x = x.clone()

            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # (B, H, W)
            x = self.symmetrize_matrix(x)
            return x

        elif self.mode == "latent_with_flash_attn_with_adalayernorm":
            for i in range(no_cycling):
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

                B, H, W, C = x.shape

                x = x.view(B, H * W, C)
                x = self.attn_encoder(x)
                x = x.view(B, H, W, C)

                t_emb = self.recycle_embed(torch.tensor([i], dtype=torch.long, device=x.device))
                x = x + self.no_cycling_linear(self.no_cycling_norm(x, t_emb))

            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # (B, H, W)
            x = self.symmetrize_matrix(x)
            return x

        elif self.mode == "latent_with_flash_attn_with_5layernorm":
            for i in range(no_cycling):
                x = self.no_cycling_norm[i](x)

                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

                B, H, W, C = x.shape

                x = x.view(B, H * W, C)
                x = self.attn_encoder(x)
                x = x.view(B, H, W, C)

                x = x + self.no_cycling_linear(x)

            x = x.permute(0, 3, 1, 2)  # (B,C,H,W)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # (B,H,W)
            x = self.symmetrize_matrix(x)
            return x
        
        elif self.mode == "latent_with_flash_attn_with_dit_layernorm":
            for i in range(no_cycling):
                t_emb = self.recycle_embed(torch.tensor([i], dtype=torch.long, device=x.device))

                resnet_scale, resnet_shift, resnet_gate, attn_scale, attn_shift, attn_gate = self.linear(t_emb).chunk(6, dim=-1)  # (B, C), (B, C)
                
                resnet_scale = resnet_scale.unsqueeze(1).unsqueeze(1)
                resnet_shift = resnet_shift.unsqueeze(1).unsqueeze(1)
                resnet_gate = resnet_gate.unsqueeze(1).unsqueeze(1)

                attn_scale = attn_scale.unsqueeze(1).unsqueeze(1)
                attn_shift = attn_shift.unsqueeze(1).unsqueeze(1)
                attn_gate = attn_gate.unsqueeze(1).unsqueeze(1)

                x1 = x.clone()
                x = self.norm_resnet(x)
                x = x * (1 + resnet_scale) + resnet_shift

                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
                x = x1 + resnet_gate * x

                x2 = x.clone()
                B, H, W, C = x.shape
                x = self.norm_attn(x)
                x = x * (1 + attn_scale) + attn_shift

                x = x.view(B, H * W, C)
                x = self.attn_encoder(x)
                x = x.view(B, H, W, C)

                x = x2 + self.no_cycling_linear(attn_gate * x)

            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # (B, H, W)
            x = self.symmetrize_matrix(x)
            return x
        
        elif self.mode == "latent_with_protenix":
            if self.lora:
                x_init = x.clone()
                x = torch.zeros_like(x_init, device=x.device)

                for step in range(no_cycling):
                    x = self.loras[step](x_init, x)

                x = x.permute(0, 3, 1, 2)
                x = self.conv_to_output(x)
                x = x.squeeze(1)
                x = self.symmetrize_matrix(x)

            else:
                if self.unet:
                    x_init = x.clone()
                    x = torch.zeros_like(x_init, device=x.device)

                    for _ in range(no_cycling):
                        x = x_init + self.no_cycling_linear(self.no_cycling_norm(x))
                        x = x.permute(0, 3, 1, 2) # Convert to (B, 64, H, W).

                        x1 = self.maxpool(x) # 64, H/2, W/2
                        x1 = self.conv1(x1) # 128, H/2, W/2

                        x2 = self.maxpool(x1) # 128, H/4, W/4
                        x2 = self.conv2(x2) # 256, H/4, W/4

                        x3 = self.maxpool(x2) # 256, H/8, W/8
                        x3 = self.conv3(x3) # 512, H/8, W/8

                        x4 = self.maxpool(x3) # 512, H/16, W/16
                        x4 = self.conv4(x4) # 1024, H/16, W/16

                        d4 = self.up4(x4) # 512, H/8, W/8
                        d4 = torch.cat((x3, d4), dim=1) # 1024, H/8, W/8
                        d4 = self.up_conv4(d4) # 512, H/8, W/8

                        d3 = self.up3(d4) # 256, H/4, W/4
                        d3 = torch.cat((x2, d3), dim=1) # 512, H/4, W/4
                        d3 = self.up_conv3(d3) # 256, H/4, W/4

                        d2 = self.up2(d3) # 128, H/2, W/2
                        d2 = torch.cat((x1, d2), dim=1) # 256, H/2, W/2
                        d2 = self.up_conv2(d2) # 128, H/2, W/2

                        d1 = self.up1(d2) # 64, H, W
                        d1 = torch.cat((x, d1), dim=1) # 128, H, W
                        x = self.up_conv1(d1) # 64, H, W

                        x = x.permute(0, 2, 3, 1)
                    
                    x = x.permute(0, 3, 1, 2)
                    x = self.conv_1x1(x)
                    x = x.squeeze(1)  # Reduce to (B, H, W).
                    x = self.symmetrize_matrix(x)

                elif self.tri_attn:
                    x_init = x.clone()
                    x = torch.zeros_like(x_init, device=x.device)

                    for _ in range(no_cycling):
                        x = x_init + self.no_cycling_linear(self.no_cycling_norm(x))

                        x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
                        x = self.resnet(x)
                        x = x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).
                        # x = x.squeeze()

                        x = self.tri_out(x)
                        x = self.tri_in(x)
                        # x = self.tri_attn_start1(x)
                        # x = self.tri_attn_end1(x)

                        # x = self.tri_attn_start2(x)
                        # x = self.tri_attn_end2(x)
                        # x = x.unsqueeze(0)

                        # B, H, W, C = x.shape
                        # x = x.view(B, H*W, C)
                        # x = self.attn_encoder(x)
                        # x = x.view(B, H, W, C)
                        
                    x = x.permute(0, 3, 1, 2)
                    x = self.conv_to_output(x)
                    x = x.squeeze(1)  # Reduce to (B, H, W).
                    x = self.symmetrize_matrix(x)

                elif self.swin_transformer:
                    x_init = x.clone()
                    x = torch.zeros_like(x_init, device=x.device)

                    for _ in range(no_cycling):
                        x = x_init + self.no_cycling_linear(self.no_cycling_norm(x))

                        x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
                        x = self.resnet(x)
                        x = x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).

                        x = self.swin_transformer(x)
                        
                    x = x.permute(0, 3, 1, 2)
                    x = self.conv_to_output(x)
                    x = x.squeeze(1)  # Reduce to (B, H, W).
                    x = self.symmetrize_matrix(x)

                else:
                    x_init = x.clone()
                    # x = torch.zeros_like(x_init, device=x.device)
                    x = self.x_init_linear(x)

                    for _ in range(no_cycling):
                        x = self.no_cycling_linear(self.no_cycling_norm(x_init + x))

                        x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
                        x = self.resnet(x)
                        x = x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).

                        # B, H, W, C = x.shape
                        # x = x.view(B, H*W, C)
                        # x = self.attn_encoder(x)
                        # x = x.view(B, H, W, C)
                        
                    x = x.permute(0, 3, 1, 2)
                    x = self.conv_to_output(x)
                    x = x.squeeze(1)  # Reduce to (B, H, W).
                    x = self.symmetrize_matrix(x)
            
            return x
        
        elif self.mode == "latent_with_protenix_with_dit_layernorm":
            x_init = x.clone()
            x = torch.zeros_like(x_init, device=x.device)

            for i in range(no_cycling):
                t_emb = self.recycle_embed(torch.tensor([i], dtype=torch.long, device=x.device))

                resnet_scale, resnet_shift, \
                attn_scale, attn_shift, \
                recycle_scale, recycle_shift = self.linear(t_emb).chunk(6, dim=-1)  # (B, C), (B, C)
                
                resnet_scale = resnet_scale.unsqueeze(1).unsqueeze(1)
                resnet_shift = resnet_shift.unsqueeze(1).unsqueeze(1)

                attn_scale = attn_scale.unsqueeze(1).unsqueeze(1)
                attn_shift = attn_shift.unsqueeze(1).unsqueeze(1)
                
                recycle_scale = recycle_scale.unsqueeze(1).unsqueeze(1)
                recycle_shift = recycle_shift.unsqueeze(1).unsqueeze(1)

                x = x_init + self.no_cycling_linear(self.norm_recycle(x) * (1 + recycle_scale) + recycle_shift)

                # x = self.norm_resnet(x) * (1 + resnet_scale) + resnet_shift

                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

                B, H, W, C = x.shape
                x = self.norm_attn(x) * (1 + attn_scale) + attn_shift

                x = x.view(B, H * W, C)
                x = self.attn_encoder(x)
                x = x.view(B, H, W, C)

            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # (B, H, W)
            x = self.symmetrize_matrix(x)
            return x

        elif self.mode == "latent_with_protenix_with_dit_layernorm_trm":
            x_init = x.clone()
            x = torch.zeros_like(x_init, device=x.device)
            y = torch.zeros_like(x_init, device=x.device)

            t_emb_k_list = [torch.tensor([i], dtype=torch.long, device=x.device) for i in range(k_no_cycling + 1)]

            # without grad
            with torch.no_grad():
                for k in range(k_no_cycling): 
                    x, y = self.latent_recursion(x_init, x, y, no_cycling, t_emb_k_list[k])

            x, y = self.latent_recursion(x_init, x, y, no_cycling, t_emb_k_list[k_no_cycling])

            # with grad
            # for k in range(k_no_cycling + 1):
            #     x, y = self.latent_recursion(x_init, x, y, no_cycling, t_emb_k_list[k])

            y = y.permute(0, 3, 1, 2)  # (B, C, H, W)
            y = self.conv_to_output(y)
            y = y.squeeze(1)  # (B, H, W)
            y = self.symmetrize_matrix(y)
            return y
        
        elif self.mode == "latent_with_protenix_with_trm":
            # x_init = x.clone()
            # x = torch.zeros_like(x_init, device=x.device)
            # y = torch.zeros_like(x_init, device=x.device)

            x_recycle = self.linear_x(x)
            y_recycle = self.linear_y(x)

            # without grad
            # with torch.no_grad():
            #     for _ in range(k_no_cycling): 
            #         x, y = self.latent_recursion_wo_time_emb(x_init, x, y, no_cycling)

            # x, y = self.latent_recursion_wo_time_emb(x_init, x, y, no_cycling)

            # with grad
            # for _ in range(k_no_cycling + 1):
            #     x, y = self.latent_recursion_wo_time_emb(x_init, x, y, no_cycling)

            # y = y.permute(0, 3, 1, 2)  # (B, C, H, W)
            # y = self.conv_to_output(y)
            # y = y.squeeze(1)  # (B, H, W)
            # y = self.symmetrize_matrix(y)

            # with grad
            for _ in range(k_no_cycling + 1):
                x_recycle, y_recycle = self.latent_recursion_wo_time_emb(x, x_recycle, y_recycle, no_cycling)

            y_recycle = y_recycle.permute(0, 3, 1, 2)  # (B, C, H, W)
            y_recycle = self.conv_to_output(y_recycle)
            y_recycle = y_recycle.squeeze(1)  # (B, H, W)
            y_recycle = self.symmetrize_matrix(y_recycle)

            return y_recycle
        
        elif self.mode == "latent_with_protenix_with_trm_3loss":
            outputs = []
            x_recycle = self.linear_x(x)
            y_recycle = self.linear_y(x)

            for _ in range(k_no_cycling + 1):
                x_recycle, y_recycle = self.latent_recursion_wo_time_emb(x, x_recycle, y_recycle, no_cycling)
                y_hat = y_recycle.permute(0, 3, 1, 2)  # (B, C, H, W)
                y_hat = self.conv_to_output(y_hat)
                y_hat = y_hat.squeeze(1)  # (B, H, W)
                y_hat = self.symmetrize_matrix(y_hat)
                outputs.append(y_hat)
            
            return outputs

        elif self.mode == "latent_with_protenix_with_ab_trm_deep_supervision":
            # print("x", x)
            # print("x_recycle", x_recycle)
            # print("y_recycle", y_recycle)

            with torch.no_grad():
                for _ in range(k_no_cycling): 
                    x_recycle, y_recycle = self.latent_recursion_ab_trm_deep_supervision(x, x_recycle, y_recycle, no_cycling)

            x_recycle, y_recycle = self.latent_recursion_ab_trm_deep_supervision(x, x_recycle, y_recycle, no_cycling)
            
            # print("x_recycle after", x_recycle)
            # print("y_recycle after", y_recycle)

            logits = y_recycle.permute(0, 3, 1, 2)  # (B, C, H, W)
            logits = self.conv_to_output(logits)
            logits = logits.squeeze(1)  # (B, H, W)

            # flag = np.all(
            #     np.isclose(logits[0].cpu().detach().numpy(), np.transpose(logits[0].cpu().detach().numpy()))
            # )
            # print("before logits shape", logits.shape)
            # print("before symmetric check:", flag)
            # print(logits[0].cpu().detach().numpy())
            logits = self.symmetrize_matrix(logits)
            
            # print("after logits shape", logits.shape)
            # flag = np.all(
            #     np.isclose(logits[0].cpu().detach().numpy(), np.transpose(logits[0].cpu().detach().numpy()))
            # )
            # print("after symmetric check:", flag)
            # print(logits[0].cpu().detach().numpy())

            return logits, x_recycle.detach(), y_recycle.detach()

        elif self.mode == "latent_with_protenix_with_trm_deep_supervision":
            # without grad
            # with torch.no_grad():
            #     for _ in range(k_no_cycling): 
            #         x_recycle, y_recycle = self.latent_recursion_wo_time_emb(x, x_recycle, y_recycle, no_cycling)

            # x_recycle, y_recycle = self.latent_recursion_wo_time_emb(x, x_recycle, y_recycle, no_cycling)

            # with grad
            for _ in range(k_no_cycling + 1):
                x_recycle, y_recycle = self.latent_recursion_wo_time_emb(x, x_recycle, y_recycle, no_cycling)

            logits = y_recycle.permute(0, 3, 1, 2)  # (B, C, H, W)
            logits = self.conv_to_output(logits)
            logits = logits.squeeze(1)  # (B, H, W)
            logits = self.symmetrize_matrix(logits)

            return logits, x_recycle.detach(), y_recycle.detach()
            # return logits, x_recycle, y_recycle

        elif self.mode == "latent_with_protenix_with_trm_deep_supervision_with_attn":
            if self.ab_no_hierarchy:
                for _ in range(no_cycling * (k_no_cycling + 1)):
                    x_recycle = self.no_cycling_linear(self.norm_recycle(x_recycle + y_recycle + x))
                    x_recycle = x_recycle.permute(0, 3, 1, 2)  # (B, C, H, W)
                    x_recycle = self.resnet(x_recycle)
                    x_recycle = x_recycle.permute(0, 2, 3, 1)  # (B, H, W, C)

                    y_recycle = self.no_cycling_linear(self.norm_recycle(x_recycle + y_recycle))

                    y_recycle = y_recycle.permute(0, 3, 1, 2)  # (B, C, H, W)
                    y_recycle = self.resnet(y_recycle)
                    y_recycle = y_recycle.permute(0, 2, 3, 1)  # (B, H, W, C)

                logits = y_recycle.permute(0, 3, 1, 2)  # (B, C, H, W)
                logits = self.conv_to_output(logits)
                logits = logits.squeeze(1)  # (B, H, W)
                logits = self.symmetrize_matrix(logits)
                return logits, x_recycle.detach(), y_recycle.detach()
            
            else:
                # without grad
                # with torch.no_grad():
                #     for _ in range(k_no_cycling): 
                #         x_recycle, y_recycle = self.latent_recursion_wo_time_emb(x, x_recycle, y_recycle, no_cycling)

                # x_recycle, y_recycle = self.latent_recursion_wo_time_emb(x, x_recycle, y_recycle, no_cycling)

                # with grad
                for _ in range(k_no_cycling + 1):
                    x_recycle, y_recycle, z_recycle = self.latent_recursion_wo_time_emb(x, x_recycle, y_recycle, z_recycle, no_cycling, sequence_x=sequence_x, sequence_y=sequence_y, z_no_cycling=z_no_cycling)

                logits = y_recycle.permute(0, 3, 1, 2)  # (B, C, H, W)

                logits = self.conv_to_output(logits)
                logits = logits.squeeze(1)  # (B, H, W)
                logits = self.symmetrize_matrix(logits)

                if self.ab_triple:
                    return logits, x_recycle.detach(), y_recycle.detach(), z_recycle.detach()
                else:
                    return logits, x_recycle.detach(), y_recycle.detach()
                # return logits, x_recycle, y_recycle

        elif self.mode == "latent_with_protenix_with_deep_supervision":
            # without grad
            # with torch.no_grad():
            #     for _ in range(no_cycling - 1): 
            #         x_recycle = x + self.no_cycling_linear(self.no_cycling_norm(x_recycle))
            #         x_recycle = x_recycle.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
            #         x_recycle = self.resnet(x_recycle)
            #         x_recycle = x_recycle.permute(0, 2, 3, 1) # Convert to (B, H, W, C).
  
            # x_recycle = x + self.no_cycling_linear(self.no_cycling_norm(x_recycle))
            # x_recycle = x_recycle.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
            # x_recycle = self.resnet(x_recycle)

            # logits = self.conv_to_output(x_recycle)
            # logits = logits.squeeze(1)  # (B, H, W)
            # logits = self.symmetrize_matrix(logits)

            # x_recycle = x_recycle.permute(0, 2, 3, 1) # Convert to (B, H, W, C).

            # with grad
            for _ in range(no_cycling):
                # x_recycle = self.no_cycling_linear(self.no_cycling_norm(x + x_recycle))
                x_recycle = self.no_cycling_linear(self.no_cycling_norm(x_recycle))
                x_recycle = x_recycle.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
                x_recycle = self.resnet(x_recycle)
                x_recycle = x_recycle.permute(0, 2, 3, 1) # Convert to (B, H, W, C).
                
            x_recycle = x_recycle.permute(0, 3, 1, 2)
            logits = self.conv_to_output(x_recycle)
            logits = logits.squeeze(1)  # Reduce to (B, H, W).
            logits = self.symmetrize_matrix(logits)
            x_recycle = x_recycle.permute(0, 2, 3, 1) # Convert to (B, H, W, C).
            
            return logits, x_recycle.detach()
        
        elif self.mode == "latent_with_protenix_tri_flash_attn":
            x_init = x.clone()
            x = torch.zeros_like(x_init, device=x.device)

            for _ in range(no_cycling):
                x = x_init + self.no_cycling_linear(self.no_cycling_norm(x))

                x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).
                x = self.resnet(x)
                x = x.permute(0, 2, 3, 1) # Convert to (B, H, W, C).

                # B, H, W, C = x.shape
                # x = x.view(B, H*W, C)
                # x = self.attn_encoder(x)
                # x = x.view(B, H, W, C)

                B, H, W, C = x.shape
                tri_mask = torch.triu(
                    torch.ones(H, W, device=x.device, dtype=torch.bool)
                )
                x_upper = x[:, tri_mask, :]
                x_upper = self.attn_encoder(x_upper)
                x_new = torch.zeros_like(x)
                x_new[:, tri_mask, :] = x_upper
                
                x_new = x_new + x_new.transpose(1, 2)

                diag_mask = torch.eye(H, device=x.device, dtype=torch.bool)
                x_new[:, diag_mask, :] = x_new[:, diag_mask, :] / 2.0

                x = x_new

            x = x.permute(0, 3, 1, 2)
            x = self.conv_to_output(x)
            x = x.squeeze(1)  # Reduce to (B, H, W).
            x = self.symmetrize_matrix(x)
            
    class ResNet2DModule(nn.Module):
        """A stack of 2D residual blocks.

        Args:
            channels (int): Number of input/output channels.
            num_res_blocks (int): Number of residual blocks.
            kernel_size (int): Kernel size for each residual block.
            use_bias (bool): Whether to use bias in convolution layers.
        """

        def __init__(
            self,
            channels: int,
            num_res_blocks: int,
            kernel_size: int = 3,
            use_bias: bool = False,
            use_sequence: bool = False
        ):
            super().__init__()
            self.use_sequence = use_sequence
            if use_sequence:
                self.down_linear = nn.Linear(2048, 32)
                self.attn_encoders = nn.ModuleList(
                    [nn.MultiheadAttention(embed_dim=2048, num_heads=8) for _ in range(2)]
                )
                
            self.res2D_blocks = nn.ModuleList(
                [
                    self.Residual2DBlock(channels, kernel_size, use_bias)
                    for _ in range(num_res_blocks)
                ]
            )

        def outer_concat(self, x):
            """Taken directly from FM4BioContactHead"""
            batch_size, seq_len, features = x.shape

            # Permute to [batch_size, features, seq_len]
            x = x.permute(0, 2, 1)

            # Introduce new dimensions for broadcasting
            x_1 = x[:, None, :, :, None]  # [batch_size, 1, features, seq_len, 1]
            x_2 = x[:, None, :, None, :]  # [batch_size, 1, features, 1, seq_len]

            # Repeat along new dimensions
            x_1 = x_1.repeat(
                1, 1, 1, 1, seq_len
            )  # [batch_size, 1, features, seq_len, seq_len]
            x_2 = x_2.repeat(
                1, 1, 1, seq_len, 1
            )  # [batch_size, 1, features, seq_len, seq_len]

            # Concatenate along the second dimension
            x = torch.cat((x_1, x_2), dim=1)  # [batch_size, 2, features, seq_len, seq_len]

            # Get lower triangular indices
            I, J = torch.tril_indices(seq_len, seq_len, -1)

            # Symmetrize
            x[:, :, :, I, J] = x[:, :, :, J, I]

            # Permute to desired shape and make contiguous
            x = x.permute(
                0, 3, 4, 2, 1
            ).contiguous()  # [batch_size, seq_len, seq_len, features, 2]

            # Reshape to combine the last two dimensions
            x = x.view(
                batch_size, seq_len, seq_len, features * 2
            )  # [batch_size, seq_len, seq_len, features * 2]

            return x
    
        def forward(self, x: Tensor, sequence: Tensor = None) -> Tensor:
            """Forward pass

            Args:
                x (Tensor): Input tensor of shape (B, C, H, W).
                sequence (Tensor, optional): Input tensor of shape (B, H, 2048).
            Returns:
                Tensor: Output tensor of the same shape as input.
            """
            if sequence is not None:
                for res_block, attn in zip(self.res2D_blocks, self.attn_encoders):
                    x = res_block(x) # 1, 64, H, W
                    sequence, _ = attn(sequence, sequence, sequence)
                    pair_sequence = self.down_linear(sequence) # linear 2048 -> 32
                    pair_sequence = self.outer_concat(pair_sequence) # (1, H, W, 64)
                    pair_sequence = pair_sequence.permute(0, 3, 1, 2) # (1, 64, H, W)
                    x = x + pair_sequence
            else:
                for res_block in self.res2D_blocks:
                    x = res_block(x)
            return x

        class Residual2DBlock(nn.Module):
            """A single residual block for a 2D ResNet.

            Args:
                channels (int): Number of input/output channels.
                kernel_size (int): Kernel size for the 2D convolution.
                use_bias (bool): Whether to use bias in convolution layers.
            """

            # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

            def __init__(self, channels, kernel_size=3, use_bias=False):
                super().__init__()
                self.residual_block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=1,
                        bias=use_bias,
                    ),
                    nn.InstanceNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        bias=use_bias,
                        padding="same",
                    ),
                    nn.InstanceNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=1,
                        bias=use_bias,
                    ),
                    nn.InstanceNorm2d(channels),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x: Tensor) -> Tensor:
                """Forward pass

                Args:
                    x (Tensor): Input tensor of shape (B, C, H, W).
                Returns:
                    Tensor: Output tensor of the same shape as input.
                """
                return x + self.residual_block(x)

    class conv_block(nn.Module):
        def __init__(self,ch_in,ch_out):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.InstanceNorm2d(ch_out),
                # nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.InstanceNorm2d(ch_out),
                # nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )


        def forward(self,x):
            x = self.conv(x)
            return x
    
    class up_conv(nn.Module):
        def __init__(self,ch_in,ch_out):
            super().__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.InstanceNorm2d(ch_out),
                # nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        def forward(self,x):
            x = self.up(x)
            return x
    
    class TriAttStart(nn.Module):
        def __init__(self,z_dim,N_head=4,c=8):
            super().__init__()
            self.z_dim = z_dim
            self.N_head = N_head
            self.c = c
            self.sq_c = 1/math.sqrt(c)
            self.norm=nn.LayerNorm(z_dim)
            self.qlinear= nn.Linear(z_dim,c*N_head)
            self.klinear= nn.Linear(z_dim,c*N_head)
            self.vlinear= nn.Linear(z_dim,c*N_head)
            self.blinear= nn.Linear(z_dim,N_head)
            self.glinear= nn.Linear(z_dim,c*N_head)
            self.olinear= nn.Linear(c*N_head,z_dim)

        def SignedSqrt(self, x):
            x = torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))
            return x
    
        def forward(self,z_):
            L1,L2,D=z_.shape
            z = self.norm(z_)
            q = self.qlinear(z).reshape(L1,L2,self.N_head,self.c)
            k = self.klinear(z).reshape(L1,L2,self.N_head,self.c)
            v = self.vlinear(z).reshape(L1,L2,self.N_head,self.c)
            b = self.blinear(z)
            att = torch.einsum('blhc,bkhc->blkh',q,k)*self.sq_c + b[None,:,:,:]
            att = F.softmax(self.SignedSqrt(att),dim=2)
            o = torch.einsum('blkh,bkhc->blhc',att,v)
            o = (torch.sigmoid(self.glinear(z).reshape(L1,L2,self.N_head,self.c)) * o).reshape(L1,L2,-1)
            o = self.olinear(o)
            return o

    class TriAttEnd(nn.Module):
        def __init__(self,z_dim,N_head=4,c=8):
            super().__init__()
            self.z_dim = z_dim
            self.N_head = N_head
            self.c = c
            self.sq_c = 1/math.sqrt(c)
            self.norm=nn.LayerNorm(z_dim)
            self.qlinear= nn.Linear(z_dim,c*N_head)
            self.klinear= nn.Linear(z_dim,c*N_head)
            self.vlinear= nn.Linear(z_dim,c*N_head)
            self.blinear= nn.Linear(z_dim,N_head)
            self.glinear= nn.Linear(z_dim,c*N_head)
            self.olinear= nn.Linear(c*N_head,z_dim)

        def SignedSqrt(self, x):
            x = torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))
            return x
        
        def forward(self,z_):
            L1,L2,D=z_.shape
            z = self.norm(z_)
            q = self.qlinear(z).reshape(L1,L2,self.N_head,self.c)
            k = self.klinear(z).reshape(L1,L2,self.N_head,self.c)
            v = self.vlinear(z).reshape(L1,L2,self.N_head,self.c)
            b = self.blinear(z)
            att = torch.einsum('blhc,kbhc->blkh',q,k)*self.sq_c + b[None,:,:,:].permute(0,2,1,3)
            att = F.softmax(self.SignedSqrt(att),dim=2)
            o = torch.einsum('blkh,klhc->blhc',att,v)
            o = (torch.sigmoid(self.glinear(z).reshape(L1,L2,self.N_head,self.c)) * o).reshape(L1,L2,-1)
            o = self.olinear(o)
            return o
    
    class TriOut(nn.Module):
        def __init__(self,z_dim,c=32):
            super().__init__()
            self.z_dim = z_dim
            self.norm =nn.LayerNorm(z_dim)
            self.onorm =nn.LayerNorm(c)
            self.alinear=nn.Linear(z_dim,c)
            self.blinear=nn.Linear(z_dim,c)
            self.aglinear=nn.Linear(z_dim,c)
            self.bglinear=nn.Linear(z_dim,c)
            self.glinear=nn.Linear(z_dim,z_dim)
            self.olinear=nn.Linear(c,z_dim)

        def forward(self,z):
            z = self.norm(z)
            a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
            b = self.blinear(z) * torch.sigmoid(self.bglinear(z))
            o = torch.einsum('bilc,bjlc->bijc',a,b)
            o = self.onorm(o)
            o = self.olinear(o)
            o = o * torch.sigmoid(self.glinear(z))
            return o

    class TriIn(nn.Module):
        def __init__(self,z_dim,c=32):
            super().__init__()
            self.z_dim = z_dim
            self.norm =nn.LayerNorm(z_dim)
            self.onorm =nn.LayerNorm(c)
            self.alinear=nn.Linear(z_dim,c)
            self.blinear= nn.Linear(z_dim,c)
            self.aglinear= nn.Linear(z_dim,c)
            self.bglinear= nn.Linear(z_dim,c)
            self.glinear = nn.Linear(z_dim,z_dim)
            self.olinear= nn.Linear(c,z_dim)

        def forward(self,z):
            z = self.norm(z)
            a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
            b = self.blinear(z) * torch.sigmoid(self.bglinear(z))
            o = torch.einsum('blic,bljc->bijc',a,b)
            o = self.onorm(o)
            o = self.olinear(o)
            o = o * torch.sigmoid(self.glinear(z))
            return o
    
    class SwinTransformer(nn.Module):
        def __init__(self, dim=64, heads=4, window_size=8):
            super().__init__()
            self.dim = dim
            self.heads = heads
            self.window_size = window_size

            self.pair_window_attention = WindowAttention(
                dim=dim,
                window_size=(window_size, window_size),
                num_heads=heads,
                qkv_bias=True,
                attn_drop=0.0,
                proj_drop=0.0,
                pretrained_window_size=(window_size, window_size)
            )
            
            self.pair_attn_ln = RnaRMSNorm(dim)

        def forward(self, x):
            shortcut = x 
            B, H, W, C = x.shape
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size

            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = F.pad(x, (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            _, Hp, Wp, _ = x.shape

            # handle mask
            pair_mask = torch.ones((B, H, W), device=x.device, dtype=x.dtype)
            if pad_b > 0 or pad_r > 0:
                pair_mask = F.pad(pair_mask, (0, pad_r, 0, pad_b))

            x_windows = window_partition(x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size*self.window_size, C)

            pair_mask_windows = window_partition(pair_mask.unsqueeze(-1).float(), self.window_size)  # [nW*B, ws, ws, 1]
            pair_mask_windows = pair_mask_windows.view(-1, self.window_size*self.window_size)
            attn_mask_win = pair_mask_windows.unsqueeze(1) - pair_mask_windows.unsqueeze(2)
            attn_mask_win = attn_mask_win.masked_fill(attn_mask_win != 0, float(-100.0)).masked_fill(attn_mask_win == 0, float(0.0))
            
            attn_windows = self.pair_window_attention(x_windows, mask=attn_mask_win)  # [nW*B, ws*ws, C]
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, Hp, Wp, C]
            
            if pad_b > 0 or pad_r > 0:
                x = x[:, :H, :W, :].contiguous()

            x = self.pair_attn_ln(x)
            x = shortcut + x
            return x

class ResNet1DAdapter(nn.Module, SequenceAdapter):
    """Adapter module that applies a ResNet1DModule to sequence data.

    Args:
        input_dim (int): Input feature dimension.
        channels (int): Number of channels for ResNet.
        num_blocks (int): Number of residual blocks in ResNet1DModule.
        dropout (float): Dropout rate.
    """

    # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

    def __init__(
        self, input_channels, num_outputs=1, channels=256, num_blocks=9, dropout=0.2
    ):
        super().__init__()
        self.in_to_conv = nn.Linear(input_channels, channels)
        self.resnet = self.ResNet1DModule(channels, num_blocks)
        self.dropout = nn.Dropout(p=dropout)
        if num_outputs != 1:
            raise NotImplementedError("Please set num_outputs = 1.")
        self.conv_to_output = nn.Linear(channels, num_outputs)

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): Input tensor of shape (B, L, C).
            padding_mask (Tensor, optional): Padding mask of shape (B, L). Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B,).
        """
        # Project input to embedding dimension
        x = self.in_to_conv(x)

        # Reshape for convolutional processing
        x = x.permute(0, 2, 1)
        x = self.resnet(x)
        x = x.permute(0, 2, 1)

        # Global pooling across the sequence length
        if padding_mask is not None:
            x[padding_mask, :] = 0.0
            x = x.sum(dim=-2) / (~padding_mask).sum(dim=-1)[:, None]
        else:
            x = x.mean(dim=-2)

        # Apply dropout and project to output
        x = self.dropout(x)
        x = self.conv_to_output(x).squeeze(-1)

        return x

    class ResNet1DModule(nn.Module):
        """A stack of 1D residual blocks.

        Args:
            channels (int): Number of input/output channels.
            num_blocks (int): Number of residual blocks.
            kernel_size (int): Size of the convolution kernel.
            use_bias (bool): Whether to use bias in convolution layers.
        """

        # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

        def __init__(self, channels, num_blocks, kernel_size=3, use_bias=False):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    self.Residual1DBlock(channels, kernel_size, use_bias=use_bias)
                    for _ in range(num_blocks)
                ]
            )

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass

            Args:
                x (Tensor): Input tensor of shape (B, C, L).
            Returns:
                Tensor: Output tensor of the same shape as input.
            """
            for block in self.blocks:
                x = block(x)
            return x

        class Residual1DBlock(nn.Module):
            """A single residual block for a 1D ResNet.

            Args:
                channels (int): Number of input/output channels.
                kernel_size (int): Size of the convolution kernel.
                stride (int): Stride for the convolution.
                use_bias (bool): Whether to use bias in convolution layers.
            """

            # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

            def __init__(self, channels, kernel_size=3, stride=1, use_bias=False):
                super().__init__()
                self.residual_block = nn.Sequential(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        bias=use_bias,
                        padding="same",
                    ),
                    nn.InstanceNorm1d(channels),
                    nn.ELU(inplace=True),
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        bias=use_bias,
                        padding="same",
                    ),
                    nn.InstanceNorm1d(channels),
                    nn.ELU(inplace=True),
                )

            def forward(self, x: Tensor) -> Tensor:
                """
                Forward pass for the residual block.
                Args:
                    x (Tensor): Input tensor of shape (B, C, L).
                Returns:
                    Tensor: Output tensor of the same shape as input.
                """
                return x + self.residual_block(x)