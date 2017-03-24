import argparse
import os.path
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import mnist

FLAGS = None

def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))


def run_training():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    with tf.Graph().as_default():
        # 初始化输入占位符
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        # 构建神经网络
        logits = mnist.inference(images_placeholder,FLAGS.hidden1,FLAGS.hidden2)
        # 添加损失函数
        loss = mnist.loss(logits, labels_placeholder)
        # 训练
        train_op = mnist.training(loss, FLAGS.learning_rate)
        # 正确率计算
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        start_time = time.time()

        for step in range(FLAGS.max_steps):

            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            if step % 100 == 0:
                duration = time.time() - start_time
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                start_time = time.time()

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
                print('Test Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    FLAGS, unparsed = parser.parse_known_args()
    run_training()