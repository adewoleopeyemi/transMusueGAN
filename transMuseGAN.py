import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras import Input
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop
import wave
import numpy as np


def preprocess_wave(data):
  wave_channels= data.getnchannels()
  wave_framerates = data.getframerate()
  total_frames = data.getnframes()
  wave_duration = 2
  wave = data.readframes(-1)
  wav_data = np.fromstring(wave, "int32")
  print(len(wav_data))
  wav_data = wav_data[:9349200]
  wav_data = np.add(wav_data, 2147483433)
  wav_data = np.divide(wav_data, 2147483433)
  print(wav_data.min)
  n_batch_size = 53
  wav_data = np.expand_dims(wav_data, axis=0)
  wav_data = wav_data.reshape(n_batch_size, 2, wave_framerates, wave_channels)
  return wav_data


def conv_t(x, f, k, s, a, p, bn):
  x = layers.Conv2DTranspose(
      filters=f,
      kernel_size=k,
      padding=p,
      strides=s,
  )(x)
  if bn:
    x = layers.BatchNormalization(momentum=0.9)(x)
  if a == "relu":
    x = layers.Activation(a)(x)
  elif a == "lrelu":
    x = layers.LeakyReLU()(x)

  return x

def TemporalNetwork():
  input_layer= Input(shape=(100,), name = "temporal_input")
  x = layers.Reshape([1,1,100])(input_layer)
  x = conv_t(x, f=256, k=(2,1), s=(1, 1), a="relu", p="valid", bn=True)
  x = conv_t(x, f=100, k=(1, 1), s=(1, 1),a="relu", p="valid", bn=True)
  output_layer = layers.Reshape([2, 100])(x)
  Model = models.Model
  return Model(input_layer, output_layer)

def linear(n_track):
  input_layer= Input(shape=(100,), name = "temporal_input")
  output_layer = layers.Dense(100, activation="relu")(input_layer)
  Model = model.Model
  return Model(input_layer, output_layer)

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=3):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def BarGenerator():
  input_layer = layers.Input(shape=(400,))
  transformer_block = TransformerBlock(400, 2, 32)
  x = transformer_block(input_layer)
  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dropout(0.1)(x)
  x = layers.Dense(88200*2, activation="tanh")(x)
  output_layer = layers.Reshape((1, 2, 44100, 2))(x)




chords_input = Input(shape=(100,), name="chords_input")
style_input = Input(shape=(100,), name="style_input")
melody_input = Input(shape=(4,100), name="melody_input")
groove_input = Input(shape=(4,100), name="groove_input")


chords_tempNetwork = TemporalNetwork()
chords_over_time = chords_tempNetwork(chords_input)

melody_over_time = [None]*4

melody_tempNetwork = [None] * 4
for track in range(4):
  melody_tempNetwork[track] = TemporalNetwork()
  melody_track = layers.Lambda(lambda x: x[:,track,:])(melody_input)
  melody_over_time[track] = melody_tempNetwork[track](melody_track)

barGen = [None] * 4
for track in range(4):
  barGen[track] = BarGenerator()


bars_output = [None] * 2
for bar in range(2):
  track_output = [None] * 4
  c = layers.Lambda(lambda x: x[:,bar,:], name = 'chords_input_bar_' + str(bar))(chords_over_time)
  s = style_input
  for track in range(4):
    m = layers.Lambda(lambda x: x[:,bar,:])(melody_over_time[track])
    g = layers.Lambda(lambda x: x[:,track,:])(groove_input)
    z_input = layers.Concatenate(axis = 1
    , name = 'total_input_bar_{}_track_{}'.format(bar, track)
    )([c,s,m,g])
    track_output[track] = barGen[track](z_input)
  bars_output[bar] = layers.Add()(track_output)
generator_output = layers.Add()(bars_output)

generator = models.Model([chords_input, style_input, melody_input, groove_input]
, generator_output)


def conv(x, f, k, s, a, p):
  x = layers.Conv2D(
      filters=f, 
      kernel_size=k, 
      padding=p,
      strides=s,
  )(x)

  if a == "relu":
    x = layers.Activation(a)(x)
  elif a=="lrelu":
    x = layers.LeakyReLU()(x)

  return x

critic_input = Input(shape=(2, 44100, 2), name='critic_input')
x = critic_input
x = conv(x, f=128, k = (1,1), s = (1,1), a = 'relu', p = 'valid')
x = conv(x, f=128, k = (2,2), s = (2,2), a = 'relu', p = 'same')
x = conv(x, f=512, k = (3,1), s = (2,1), a = 'relu', p = 'same')
x = layers.Flatten()(x)
critic_output = layers.Dense(1, activation=None)(x)

critic = models.Model(critic_input, critic_output)


critic.trainable = False
chords_input = Input(shape=(100,), name="chords_input")
style_input = Input(shape=(100,), name="style_input")
melody_input = Input(shape=(4,100), name="melody_input")
groove_input = Input(shape=(4,100), name="groove_input")
model_output = critic(generator((chords_input, style_input, melody_input, groove_input)))
model = models.Model([chords_input, style_input, melody_input, groove_input], model_output)

def wasserstein(y_true, y_pred):
  return -K.mean(y_true * y_pred)
critic.compile(
optimizer= RMSprop(lr=0.00005)
, loss = wasserstein
)
model.compile(
optimizer= RMSprop(lr=0.00005)
, loss = wasserstein
)
def train_critic(x_train, batch_size, clip_threshold):
  valid = np.ones((batch_size,1))
  fake = np.zeros((batch_size,1))
  # TRAIN ON REAL IMAGES
  idx = np.random.randint(0, x_train.shape[0], batch_size)
  true_imgs = x_train[idx]
  critic.train_on_batch(true_imgs, valid)
  # TRAIN ON GENERATED IMAGES
  chords_input = np.random.normal(0, 1, (batch_size, 100))
  style_input=np.random.normal(0, 1, (batch_size, 100))
  melody_input=np.random.normal(0, 1, (batch_size, 4, 100))
  groove_input=np.random.normal(0, 1, (batch_size, 4, 100))
  gen_imgs = generator.predict([chords_input,style_input,melody_input,groove_input])
  critic.train_on_batch(gen_imgs, fake)
  for l in critic.layers:
    weights = l.get_weights()
    weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
    l.set_weights(weights)

def train_generator(batch_size):
  valid = np.ones((batch_size,1))
  chords_input = np.random.normal(0, 1, (batch_size, 100))
  style_input=np.random.normal(0, 1, (batch_size, 100))
  melody_input=np.random.normal(0, 1, (batch_size, 4, 100))
  groove_input=np.random.normal(0, 1, (batch_size, 4, 100))
  model.train_on_batch((chords_input,style_input,melody_input,groove_input), valid)
    
def train(epochs, batch_size, clip_threshold):
    epochs = epochs
    for epoch in range(epochs):
      for i in range(30):
        train_critic(x_train, batch_size = batch_size, clip_threshold = clip_threshold)

      print(f"training generator...epoch{epoch}".format(epoch))
      for _ in range(20):
        train_generator(batch_size)