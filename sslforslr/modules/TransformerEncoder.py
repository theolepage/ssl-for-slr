import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dropout, Dense, LayerNormalization, MultiHeadAttention
from tensorflow_addons.layers import GELU

class TransformerEncoderLayer(Model):
    '''
    Self-attention layer composing TransformerEncoder model.
    '''

    def __init__(self, config):
        super().__init__()

        self.activation_fn = GELU()

        self.self_attn = MultiHeadAttention(
            config.transformer_attention_heads,
            config.transformer_dim,
            dropout=config.dropout
        )

        self.dropout1 = Dropout(config.dropout)
        self.dropout2 = Dropout(0.0)
        self.dropout3 = Dropout(config.dropout)

        self.self_attn_layer_norm = LayerNormalization()
        self.fc1 = Dense(config.transformer_ffn_dim)
        self.fc2 = Dense(config.transformer_dim)

        self.final_layer_norm = LayerNormalization()

    def call(self, X):
        residual = X

        X, attn = self.self_attn(query=X,
                                 key=X,
                                 value=X,
                                 return_attention_scores=True)

        X = self.dropout1(X)
        X = residual + X

        X = self.self_attn_layer_norm(X)

        residual = X
        X = self.activation_fn(self.fc1(X))
        X = self.dropout2(X)
        X = self.fc2(X)
        X = self.dropout3(X)
        X = residual + X
        X = self.final_layer_norm(X)

        return X

class TransformerEncoder(Model):
    '''
    Self-attention Transformer implemented as a Keras model.

    This implementation is based on fairseq MultiheadAttention module.

    "Attention Is All You Need"
    Ashish Vaswani et al.
    https://arxiv.org/pdf/1706.03762.pdf
    '''

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.pos_conv = Conv1D(
            self.config.transformer_dim,
            kernel_size=128,
            padding='same',
            groups=1 # FIXME: should be 16 but backprop does not work
        )
        self.pos_conv_activation = GELU()

        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(self.config.dropout)

        self.transformer_layers = [
            TransformerEncoderLayer(self.config)
            for _ in range(self.config.transformer_layers)
        ]

    def call(self, X):
        X_conv = self.pos_conv_activation(self.pos_conv(X))
        X = X + X_conv
        X = self.layer_norm(X)
        X = self.dropout(X)
        for layer in self.transformer_layers:
            X = layer(X)
        return X