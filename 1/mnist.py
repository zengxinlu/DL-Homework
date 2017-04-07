import math
import tensorflow as tf

NUM_CLASSES = 10

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def weight_variable(shape, stddev):
  initial = tf.truncated_normal(shape, stddev = stddev)
  return tf.Variable(initial, name = 'weights')

def bias_variable(shape, constant):
  initial = tf.constant(constant, shape = shape)
  return tf.Variable(initial, name = 'biases')

def inference(images, hidden1_units, hidden2_units):
  # conv 1
  # with tf.name_scope('conv1'):
  #   weights = weight_variable([])
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = weight_variable([IMAGE_PIXELS, hidden1_units], 1.0 / math.sqrt(float(IMAGE_PIXELS)))
    biases = bias_variable([hidden1_units], 0.1)
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = weight_variable([hidden1_units, hidden2_units], 1.0 / math.sqrt(float(hidden1_units)))
    biases = bias_variable([hidden2_units], 0.1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = weight_variable([hidden2_units, NUM_CLASSES], 1.0 / math.sqrt(float(hidden2_units)))
    biases = bias_variable([NUM_CLASSES], 0.1)
    logits = tf.matmul(hidden2, weights) + biases
  return logits


def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))