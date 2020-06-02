
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM,Dense,Bidirectional
import numpy as np

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def embedding_lookup(embedding_table,
                     input_ids,
                     input_shape,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.gather()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.gather(embedding_table, flat_input_ids)

  #input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return output

class lstm(tf.keras.Model):
    def  __init__(self, config):
        super(lstm, self).__init__(self)
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.embedding_table = tf.Variable(
            name="bert/embeddings/word_embeddings",
            trainable=False,
            dtype=tf.float32,
            initial_value=np.random.normal(0, 0.02, (self.vocab_size, 768)))
        self.seq_len = config["max_seq_len"]
        self.batch_size = config["batch_size"]
        self.num_out = 3
        self.emb_input_shape = [-1, self.seq_len, 1]
        forward_layer = LSTM(self.hidden_size, return_sequences=True, recurrent_dropout=0.1)
        backward_layer = LSTM(self.hidden_size, return_sequences=True, go_backwards=True, recurrent_dropout=0.1)
        self.rnn = Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(-1, self.seq_len, 768+self.num_out))
        self.dense_layer_1 = Dense(self.hidden_size, kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.dense_layer_2 = Dense(self.num_out, kernel_regularizer=tf.keras.regularizers.l2(0.0002))
    def predict(self, input_data, is_train=True):
        label = input_data["labels"]
        prior_info = input_data["prior_info"]
        input_ids = input_data["input_ids"]
        input_ids = tf.compat.v1.placeholder_with_default(input_ids, tf.TensorShape([None, self.seq_len]))
        prior_info = tf.compat.v1.placeholder_with_default(prior_info, tf.TensorShape([None, self.seq_len]))
        # get bert embedding
        embedding_output = []
        #with tf.GradientTape() as tape:
        embedding_output = embedding_lookup(
                  embedding_table=self.embedding_table,
                  input_ids=input_ids,
                  input_shape = self.emb_input_shape,
                  vocab_size=self.vocab_size,
                  embedding_size=768,
                  initializer_range=0.02,
                  word_embedding_name="word_embeddings",
                  use_one_hot_embeddings=False)
        # concat prior info
        prior_info = tf.one_hot(prior_info, depth=self.num_out, dtype=tf.float32)
        prior_info = tf.reshape(prior_info, [-1, self.seq_len, self.num_out])
        full_info = tf.concat([embedding_output, prior_info], -1)
        #rnn
        rnn_outputs = self.rnn(full_info, training=is_train)
        hidden_layer_1 = rnn_outputs
        hidden_layer_1 = tf.reshape(hidden_layer_1, [-1, self.hidden_size*2])
        hidden_layer_2 = self.dense_layer_1(hidden_layer_1)
        res = self.dense_layer_2(hidden_layer_2)
        res = tf.reshape(res, [-1, self.seq_len, self.num_out])
        #softmax
        probabilities = tf.nn.softmax(res, axis=-1)
        #print(probabilities.name)
        # calculate loss
        log_prob = tf.nn.log_softmax(res, axis=-1)
        label = tf.one_hot(label, depth=self.num_out, dtype=tf.float32)
        examples_loss = -tf.reduce_sum(tf.reduce_sum(log_prob*tf.cast(label, tf.float32), axis=-1), axis=-1)
        loss = tf.reduce_mean(examples_loss)
        return  probabilities, loss
    def call(self, x):
        return self.predict(x)
