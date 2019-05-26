from __future__ import print_function
from time import time
import datetime
import argparse
import gzip
import json
import os
import pickle
import numpy as np
import tensorflow as tf


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)


    '''training and validatingn accuracy'''
def train_validate(X_train, Y_onehot_train, X_valid, Y_onehot_valid, num_epochs, lr, num_filters, batch_size, filter_size):
    
    X_train, Y_onehot_train, X_valid, Y_onehot_valid, X_test, Y_onehot_test = mnist()
    # Fully-connected layer.
    fc_size = 128             # Number of neurons in fully-connected layer.
    img_size = 28             # pixel size
    img_size_flat = img_size*img_size   # 784

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)
    # Number of classes, one class for each of 10 digits.
    num_classes = 10
    
    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = 1
    
    '''PLACEHOLDERS'''
    x_image = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x_image')
    y_label = tf.placeholder(tf.int64, shape=[None, num_classes], name='y_label')
    
# Convolutional Layer #1 28x28x1 with 3x3*1 with 16 filters gives 28x28x16
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
# Pooling Layer #1 gives us an output of 27x27x1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)

# Convolutional Layer2 takes input from pooling layer 1 and outputs 27x27x1
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
# Pooling Layer #2 gives us an output of 26x26x1    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)

#Flattening the layer 
    layer_flat = tf.reshape(pool2, [-1, 26 * 26 * 16])
    
#Fully connected layer with 128 units  
    layer_fc1 = tf.layers.dense(inputs=layer_flat, units=128, activation=tf.nn.relu)

#Fully connected layer with 10 units as our output
    y = tf.layers.dense(inputs=layer_fc1, units=10)
    
#print('y predict', y)
    acc_train = []
    acc_valid = []
#calculating the cost by taking mean of loss    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels = y_label), name='cost')
#gradient descent optimizer
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
#number of correct predictions
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_label,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    model = tf.train.Saver()
#tensorflow 1st session    
    with tf.Session() as sess1:
      sess1.run(tf.global_variables_initializer())
      start_time = time()
      for epoch in range(num_epochs):
        correctsum = 0
        x_train_size_batch = len(X_train)//batch_size #int(X_train.shape[0]/x_batch.shape[0])
#accuracy over training 
        for i  in range(x_train_size_batch):
            x_batch = X_train[i*batch_size:(i+1)*batch_size]  #sequential picking up of batches
            y_batch = Y_onehot_train[i*batch_size:(i+1)*batch_size]
            
            '''random shuffling of data not giving increase in accuracy with increase in epochs'''
      
            #x_batch = tf.random_shuffle(X_train)[:batch_size]   
            #y_batch = tf.random_shuffle(Y_onehot_train)[:batch_size]
            
            _, loss = sess1.run([optimizer, cost],feed_dict={x_image:x_batch, y_label: y_batch})
            batch_correct_count = sess1.run(accuracy,feed_dict={x_image:x_batch, y_label: y_batch})
            correctsum += batch_correct_count   # averaging over total batch count 

        total_accuracy_train = correctsum/x_train_size_batch  

        acc_train.append(total_accuracy_train)

        print('epoch {}, loss {:.4f} train accuracy {:.2f}%'.format(epoch, loss, total_accuracy_train*100), end='\n')

        model.save(sess1, 'final_model')  #MODEL getting saved
        
        correctsum_val = 0

#accuracy over validation
        
        x_val_size_batch = len(X_valid)//batch_size  #int(X_valid.shape[0]/x_val.shape[0])
        for i  in range(x_val_size_batch):
            x_val = X_valid[i*batch_size:(i+1)*batch_size]
            y_val = Y_onehot_valid[i*batch_size:(i+1)*batch_size]
            batch_correct_count = sess1.run(accuracy,feed_dict={x_image:x_val, y_label: y_val})
            correctsum_val += batch_correct_count     # averaging over total batch count 
        total_accuracy_valid = correctsum_val/x_val_size_batch
        acc_valid.append(total_accuracy_valid)
        print('epoch {}, validation accuracy {:.2f}%'.format(epoch, total_accuracy_valid*100), end='\n')
      
      print('final model saved')

    return acc_valid, model

def test_mnist(X_test, Y_onehot_test):    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('final_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print('Model Restored')
        graph = tf.get_default_graph()
        x_image = graph.get_tensor_by_name('x_image:0')
        y_label = graph.get_tensor_by_name('y_label:0')
        feed_dict = {x_image: X_test, y_label: Y_onehot_test}
        cost = graph.get_tensor_by_name('cost:0')
        test_error = sess.run(cost, feed_dict)
    return test_error

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-1, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=64, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
                        help="Filter width and height")
    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size
  
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)
    
    learning_curve, model = train_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)
    test_error = test_mnist(x_test, y_test)
    tf.reset_default_graph()
    print('validation accuracy is', learning_curve)
    ''' for loop for calculating at different learning rates'''
    #for lr in [0.1, 0.01, 0.001, 0.0001]:
        #learning_curve, model = train_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)
        #print(learning_curve)


    ''' for loop for calculating at different filter sizes'''
    #for lr in [1, 3, 5, 7]:
        #learning_curve, model = train_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)
        #print(learning_curve) 
    
# saving results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = [learning_curve]
    results["test_error"] = test_error.tolist()

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    with open(fname, "w") as fh:
    	json.dump(results, fh)
    fh.close()

