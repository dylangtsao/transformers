import tensorflow as tf

# Hyperparameters
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 10000  # example value, change based on your dataset
target_vocab_size = 10    # number of entity tags + 1 for 'O' (Outside any entity)
dropout_rate = 0.1
max_position = 200  # maximum length of input sequence

# Transformer Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=dropout_rate):
        super(EncoderLayer, self).__init__()

        self.mha = tf.MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# NER Transformer
class NERTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, max_position, rate=dropout_rate):
        super(NERTransformer, self).__init__()

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(num_layers):
            x = self.enc_layers[i](x, training, mask)

        logits = self.final_layer(x)
        return logits

# Instantiate the NER Transformer
model = NERTransformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_position)
input_sequence = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=200)  # batch_size=64, sequence_length=50
output = model(input_sequence, training=False, mask=None)
print(output.shape)  # (batch_size, sequence_length, target_vocab_size)
