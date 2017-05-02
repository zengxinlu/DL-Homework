from __future__ import division, print_function, absolute_import  
  
import tflearn  
from tflearn.layers.core import input_data, dropout, fully_connected  
from tflearn.layers.conv import conv_2d, max_pool_2d  
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression  
  
import tflearn.datasets.oxflower17 as oxflower17  
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))  
  
# Building 'AlexNet'  
network = input_data(shape=[None, 227, 227, 3])  
network = conv_2d(network, 96, 11, strides=4, activation='relu', padding='valid')  
# network = local_response_normalization(network)  
network = batch_normalization(network)
network = max_pool_2d(network, 3, strides=2, padding='valid')
# network = local_response_normalization(network)   

network = conv_2d(network, 256, 5, activation='relu')  
# network = local_response_normalization(network)
network = batch_normalization(network)  
network = max_pool_2d(network, 3, strides=2, padding='valid')
# network = local_response_normalization(network)   

network = conv_2d(network, 384, 3, activation='relu')  

network = conv_2d(network, 384, 3, activation='relu') 
# network = dropout(network, 0.75)   

network = conv_2d(network, 256, 3, activation='relu')  
# network = local_response_normalization(network) 
network = max_pool_2d(network, 3, strides=2, padding='valid') 
# network = local_response_normalization(network)  
network = dropout(network, 0.5)   

network = fully_connected(network, 4096, activation='relu')  
network = dropout(network, 0.5)

network = fully_connected(network, 4096, activation='relu')  
network = dropout(network, 0.5)  

# network = fully_connected(network, 256, activation='relu')  

network = fully_connected(network, 17, activation='softmax')  

network = regression(network, optimizer='adam',  
                     loss='categorical_crossentropy',  
                    #  learning_rate=0.001)
                     learning_rate=0.0001)  
  
# Training  
model = tflearn.DNN(network, checkpoint_path='model_alexnet-change-none-dropout-cjq2',  
                    max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir='.\\log')  
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,  
          show_metric=True, batch_size=64, snapshot_step=200,  
          snapshot_epoch=True, run_id='alexnet-change-none-dropout-cjq2')