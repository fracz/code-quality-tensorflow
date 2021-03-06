import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import csv
import math
from random import shuffle
import sys

####################### PARAMS
vocabulary_size = 129 # liczba tokenow w kodzie
embedding_size = 100 # rozmiar wektora wejsciowego (arbitralny chyba?)
num_hidden = 128
num_classes = 2
training_steps = 50000
batch_size = 64 #128
display_step = 20
restore = False


################# DATA INPUT

datasetName = sys.argv[1]
savePath = './trained/model3/{}'.format(datasetName)
if not os.path.exists(savePath):
    os.makedirs(savePath)

class RefactorDataset():
    def __init__(self):
        self.data = np.genfromtxt('./input/{}/input.csv'.format(datasetName), delimiter=',')
        self.labels = np.genfromtxt('./input/{}/labels.csv'.format(datasetName), delimiter=',')
        self.seqlen = np.genfromtxt('./input/{}/lengths.csv'.format(datasetName), delimiter=',')
        self.max_seqlen = len(self.data[0])
        self.test_len = math.floor(len(self.data) * .15)
        self.batch_id = self.test_len
        print("Input size: " + str(self.max_seqlen) + 'x' + str(len(self.data)))

    def validation(self, batch_size):
        return self.data[0:batch_size], self.labels[0:batch_size], self.seqlen[0:batch_size]

    def test(self):
        return self.data[0:self.test_len], self.labels[0:self.test_len], self.seqlen[0:self.test_len]

    def next(self, batch_size):
        # Return a batch of data. When dataset end is reached, start over.
        if self.batch_id == len(self.data):
            self.batch_id = self.test_len
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

dataset = RefactorDataset()

############################################ RNN

device_name = sys.argv[1]

#with tf.device("/" + device_name):

train_inputs = tf.placeholder(tf.int32, shape=[None, dataset.max_seqlen])
train_outputs = tf.placeholder(tf.float32, shape=[None, num_classes])
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
seqlen = tf.placeholder(tf.int32, [None])

# https://www.tensorflow.org/tutorials/word2vec#building_the_graph
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def BiRNN(x, seqlen, weights, biases):
    inputs = tf.unstack(x, num=dataset.max_seqlen, axis=1)
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32, sequence_length=seqlen)
    outputs = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])
    print(outputs.shape)
    #outputs = tf.reduce_max(outputs, axis=1)
    print(outputs.shape)
    #return tf.matmul(outputs[-1], weights['out']) + biases['out']
    z = tf.layers.dense(outputs, 2)
    return z
    #return tf.matmul(outputs, weights['out']) + biases['out']

def cost(prediction, target):
    target = tf.expand_dims(target,axis=1)
    print('tagret shape: {}'.format(target.shape))
    cross_entropy = -tf.reduce_mean(target * tf.log(prediction), [1, 2])
    return tf.reduce_mean(cross_entropy)

logits = BiRNN(embed, seqlen, weights, biases)
prediction = tf.nn.softmax(logits)
loss_op = cost(prediction, train_outputs)
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_outputs))
print(loss_op.shape)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
train_op = optimizer.minimize(loss_op)

print(prediction.shape)
prediction = tf.reduce_mean(prediction, axis=1)
#prediction = prediction[:, -1, :]
print(prediction.shape)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(train_outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

saver = tf.train.Saver()

config = tf.ConfigProto(
#    device_count = {'GPU': 2}
#    , log_device_placement=True
)

with tf.Session(config=config) as sess:
    if restore:
        new_saver = tf.train.import_meta_graph(savePath + '/model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(savePath))
        all_vars = tf.get_collection('vars')
        for v in all_vars:
            v_ = sess.run(v)
            print(v_)
    else:
        sess.run(init)

        test_x, test_y, test_seqlen = dataset.test()

        for step in range(1, training_steps+1):
            batch_x, batch_y, batch_seqlen = dataset.next(batch_size)

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={train_inputs: batch_x, train_outputs: batch_y, seqlen: batch_seqlen})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={train_inputs: test_x,
                                                                     train_outputs: test_y,
                                                                     seqlen: test_seqlen})
                print(str(step) + "\t" + "{:.4f}".format(loss).replace('.', ',') + "\t" + "{:.3f}".format(acc).replace('.', ','))
                sys.stdout.flush()

        print("Optimization Finished!")

        saver.save(sess, savePath + '/model')

    test_x, test_y, test_seqlen = dataset.test()

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={train_inputs: test_x,
                                                            train_outputs: test_y,
                                                            seqlen: test_seqlen}))
