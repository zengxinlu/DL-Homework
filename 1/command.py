import argparse    

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
    default=5000,
    help='Number of steps to run trainer.'
)
parser.add_argument(
    '--hidden1',
    type=int,
    default=300,
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
flags, unparsed = parser.parse_known_args()

