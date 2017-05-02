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

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inference(images, hidden1_units, hidden2_units):
  # conv 1
  x_image = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
  with tf.name_scope('conv1'):
    weights = weight_variable([5, 5, 1, 32], 0.1)
    biases = bias_variable([32], 0.1)
    conv1 = tf.nn.relu(conv2d(x_image, weights) + biases)
    pool1 = max_pool_2x2(conv1)

  # conv 2
  with tf.name_scope('conv2'):
    weights = weight_variable([5, 5, 32, 64], 0.1)
    biases = bias_variable([64], 0.1)
    conv2 = tf.nn.relu(conv2d(pool1, weights) + biases)
    pool2 = max_pool_2x2(conv2)

  # Hidden 1
  pool2Size = 7 * 7 * 64;
  pool2Flat = tf.reshape(pool2, [-1, pool2Size])
  # pool2Size = IMAGE_PIXELS
  # pool2Flat = images
  with tf.name_scope('hidden1'):
    weights = weight_variable([pool2Size, hidden1_units], 1.0 / math.sqrt(float(pool2Size)))
    biases = bias_variable([hidden1_units], 0.1)
    hidden1 = tf.nn.relu(tf.matmul(pool2Flat, weights) + biases)

  # # Hidden 2
  # with tf.name_scope('hidden2'):
  #   weights = weight_variable([hidden1_units, hidden2_units], 1.0 / math.sqrt(float(hidden1_units)))
  #   biases = bias_variable([hidden2_units], 0.1)
  #   hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  # Linear
  keep_prob = tf.placeholder(tf.float32)
  # keep_prob = 0.5
  # hidden1Drop = tf.nn.dropout(hidden1, keep_prob)
  hidden1Drop = hidden1
  with tf.name_scope('softmax_linear'):
    weights = weight_variable([hidden1_units, NUM_CLASSES], 1.0 / math.sqrt(float(hidden1_units)))
    biases = bias_variable([NUM_CLASSES], 0.1)
    logits = tf.matmul(hidden1Drop, weights) + biases
  return logits, keep_prob


def loss(logits, labels):
  labels = tf.to_int64(labels)
  print(labels, logits)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
  optimizer = tf.train.AdamOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))