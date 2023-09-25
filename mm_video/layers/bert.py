# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 13:57
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : bert.py

import torch
import torch.nn as nn
import math


def make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """
    Args:
        input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        max_v_len: int, the first `max_v_len` is for video and its padding, the length
            of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
            Note max_v_len may also include the memory len (M), thus max_v_len += M
        max_t_len: int
        memory_len: int, M
    Returns:

    >>> max_v_len = 2; max_t_len=3; input_mask = torch.randn(2, 5)
    >>> make_pad_shifted_mask(input_mask, max_v_len, max_t_len)[0]
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])
    """
    bsz, seq_len = input_mask.shape
    assert max_v_len + max_t_len + memory_len == seq_len, f"{max_v_len} {max_t_len} {memory_len} {seq_len}"
    shifted_mask = input_mask.new_zeros(bsz, max_v_len + max_t_len, seq_len)  # (N, L, M+L)
    shifted_mask[:, :, :memory_len + max_v_len] = 1
    shifted_mask[:, max_v_len:, memory_len + max_v_len:] = torch.tril(input_mask.new_ones(max_t_len, max_t_len),
                                                                      diagonal=0)
    return shifted_mask


def make_pad_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """input_mask: (N, L), """
    shifted_mask = make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=memory_len)
    kg_masks = shifted_mask * input_mask.unsqueeze(1)
    return kg_masks


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask != None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertAttention_Cross(nn.Module):
    def __init__(self, config):
        super(BertAttention_Cross, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, q, kv, attention_mask=None):
        self_output = self.self(q, kv, kv, attention_mask)
        attention_output = self.output(self_output, q)
        return attention_output


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayerNoMemory(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemory, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, n_v, n_t):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, n_v, n_t)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertSelfEncoder(nn.Module):
    def __init__(self, cap_config):
        super(BertSelfEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(cap_config.vocab_size, cap_config.word_vec_size, padding_idx=0)  # 词嵌入
        self.word_fc = nn.Sequential(  # 300->768
            BertLayerNorm(cap_config.word_vec_size, eps=cap_config.layer_norm_eps),
            nn.Dropout(cap_config.hidden_dropout_prob),
            nn.Linear(cap_config.word_vec_size, cap_config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(cap_config.hidden_size, eps=cap_config.layer_norm_eps),
        )
        self.video_embeddings = nn.Sequential(
            BertLayerNorm(cap_config.video_feature_size, eps=cap_config.layer_norm_eps),
            nn.Dropout(cap_config.hidden_dropout_prob),
            nn.Linear(cap_config.video_feature_size, cap_config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(cap_config.hidden_size, eps=cap_config.layer_norm_eps),
        )
        self.position_embeddings = PositionEncoding(n_filters=cap_config.hidden_size, max_len=1000)
        self.token_type_embeddings = nn.Embedding(3, cap_config.hidden_size)
        self.LayerNorm = BertLayerNorm(cap_config.hidden_size, eps=cap_config.layer_norm_eps)
        self.dropout = nn.Dropout(cap_config.hidden_dropout_prob)

        self.config = cap_config
        self.layers = cap_config.num_hidden_layers
        self.layer_sa = nn.ModuleList([BertLayerNoMemory(cap_config) for _ in range(self.layers)])

    def forward(self, vhidden_states, thidden_states, attention_mask, type_ids, encoded=False):
        if not encoded:
            # thidden_states could be encoded (pre-encoded using another text encoder), if not, add word embedding
            thidden_states = self.word_embeddings(thidden_states)
        thidden_states = self.word_fc(thidden_states)

        vhidden_states = self.video_embeddings(vhidden_states)
        hidden_states = torch.cat((vhidden_states, thidden_states), dim=1)

        token_type_embeddings = self.token_type_embeddings(type_ids)
        hidden_states = hidden_states + token_type_embeddings
        hidden_states = self.position_embeddings(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for i in range(self.layers):
            hidden_states = self.layer_sa[i](hidden_states, attention_mask, self.config.max_v_len,
                                             self.config.max_t_len)
        return hidden_states


class BertLayerNoMemory_Cross(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemory_Cross, self).__init__()
        self.config = config
        self.attention = BertAttention_Cross(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, q, kv):
        # shifted_self_mask = make_pad_cross_mask()  # (N, L, L)
        attention_output = self.attention(q, kv)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertCrosEncoder(nn.Module):
    def __init__(self, cap_config):
        super(BertCrosEncoder, self).__init__()
        self.word_fc = nn.Sequential(  # 300->768
            BertLayerNorm(cap_config.word_vec_size, eps=cap_config.layer_norm_eps),
            nn.Dropout(cap_config.hidden_dropout_prob),
            nn.Linear(cap_config.word_vec_size, cap_config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(cap_config.hidden_size, eps=cap_config.layer_norm_eps),
        )
        self.video_embeddings = nn.Sequential(
            BertLayerNorm(cap_config.video_feature_size, eps=cap_config.layer_norm_eps),
            nn.Dropout(cap_config.hidden_dropout_prob),
            nn.Linear(cap_config.video_feature_size, cap_config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(cap_config.hidden_size, eps=cap_config.layer_norm_eps),
        )
        self.layers = cap_config.num_hidden_layers
        self.layer_sa = nn.ModuleList([BertLayerNoMemory(cap_config) for _ in range(self.layers)])
        self.layer_ca = nn.ModuleList([BertLayerNoMemory_Cross(cap_config) for _ in range(self.layers)])
        self.max_t_len = cap_config.max_t_len

    def forward(self, vhidden_states, thidden_states, tmask, type_ids):
        thidden_states = self.word_fc(thidden_states)
        vhidden_states = self.video_embeddings(vhidden_states)
        for i in range(self.layers):
            # thidden_states = self.layer_sa[i](thidden_states, tmask.int(), 0, self.max_t_len) 加上效果变差
            thidden_states = self.layer_ca[i](thidden_states, vhidden_states)
        return thidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None, \
                "bert_model_embedding_weights should not be None " \
                "when setting --share_wd_cls_weight flag to be true"
            assert config.hidden_size == bert_model_embedding_weights.size(1), \
                "hidden size has be the same as word embedding size when " \
                "sharing word embedding weight and classifier weight"
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                     bert_model_embedding_weights.size(0),
                                     bias=False)
            self.decoder.weight = torch.nn.Parameter(bert_model_embedding_weights.clone())
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x
