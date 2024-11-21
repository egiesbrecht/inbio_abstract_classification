import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel

from utils import get_act_func, GenOutput
from rmsnorm import RMSNorm
from rotary_embeddings import RotaryEmbedding


class AAConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_layers=1,
        hidden_act="silu",
        att_act="relu",
        max_position_embeddings=512,
        dropout_prob=0.1,
        initializer_range=0.02,
        rms_norm_eps=1e-8,
        group_norm_eps=1e-8,
        layer_norm_eps=1e-12,
        num_norm_groups=12,
        num_norm_channels=12,
        vocab_size=50_257,
        num_labels=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.hidden_act = hidden_act
        self.att_act = att_act
        self.num_labels = num_labels
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.group_norm_eps = group_norm_eps
        self.layer_norm_eps = layer_norm_eps
        self.num_norm_groups = num_norm_groups
        self.num_norm_channels = num_norm_channels
        self.vocab_size = vocab_size


class AAPreTrainedModel(PreTrainedModel):
    config_class = AAConfig
    base_model_prefix = "ActivatedAttentionEncoder"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ActivatedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = dim = config.hidden_size
        self.in_proj = nn.Linear(dim, dim * 3)
        #self.out_proj = nn.Linear(dim, dim)
        self.act = get_act_func(config.att_act)
        self.rope = RotaryEmbedding(dim)
        self.group_norm = nn.GroupNorm(config.num_norm_groups, config.num_norm_channels, eps=config.group_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

    def _qkv(self, x):
        Q, K, V = (self.in_proj(x)).split((self.dim, self.dim, self.dim), -1)
        
        Q = self.rope.rotate_queries_or_keys(Q)
        K = self.rope.rotate_queries_or_keys(K)
        
        Q = self.act(Q)
        K = self.act(K)
        V = self.act(V)
        
        return Q, K, V

    def _out(self, x):
        B, T, C = x.shape
        x = self.dropout(x)
        y = self.group_norm(x.reshape(-1, self.config.num_norm_channels)).reshape(x.shape)
        return y.transpose(-2, -1).reshape(B, T, self.dim)

    def forward(self, x, att_mask=None):
        B, T, C = x.shape
        Q, K, V = self._qkv(x)

        A = (Q @ K.transpose(-2, -1)) #/ (self.dim ** .5)
        # if att_mask is not None:
        #     A *= att_mask
        y = A @ V
        return self._out(y)


class ActivatedAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rms = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.att = ActivatedAttention(config)

    def forward(self, x, att_mask=None):
        y = self.att(x, att_mask) + x
        return self.layer_norm(y)


class ActivatedAttentionEncoder(AAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            ActivatedAttentionLayer(config) for _ in range(config.num_layers)
        ])

    def forward(self, x, att_mask=None):
        for L in self.layers:
            x = L(x, att_mask)
        return x


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = get_act_func(config.hidden_act)

    def forward(self, x):
        y = self.proj(x[:, 0])
        return self.act(y)


class ActivatedAttentionForSequenceClassification(AAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_size),

        )
        self.encoder = ActivatedAttentionEncoder(config)
        self.pooler = Pooler(config)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss() if config.num_labels > 2 else nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, attention_mask, labels):
        emb = self.embeddings(input_ids)
        logits = self.encoder(emb)
        out = self.cls(self.pooler(logits))
        loss = self.loss_fn(out, labels)
        return GenOutput(
            logits=out,
            loss=loss
        )
