import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("./MNIST_data/",one_hot = True)

trainimg = mnist.train.images  # training image
trainlabel = mnist.train.labels # training labels
testimg = mnist.test.images # test image
testlabel = mnist.test.labels # test lables

x = tf.placeholder(tf.float32,[None,784],name = "x")

y = tf.placeholder(tf.float32,[None,10],name = "y")

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

actv = tf.nn.softmax(tf.matmul(x,W) + b,name = "func")

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(actv,1e-10,1.0)),reduction_indices=1))

learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

pred = tf.equal(tf.argmax(actv,1),tf.argmax(y,1))

accr = tf.reduce_mean(tf.cast(pred,tf.float32))

init_op = tf.global_variables_initializer()

training_epochs = 50
batch_size = 100
display_step = 5

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, "./MNIST_model/MNIST.model")
    for epoch in range(training_epochs):
        avg_cost = 0
        num_batch = int(mnist.train.num_examples / batch_size)

        for i in range(num_batch):
            #batch_xs is shape of [100,784],batch_ys is shape of [100,10]
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            feeds = {x: batch_xs, y: batch_ys}
            avg_cost += sess.run(cost, feed_dict=feeds) / num_batch

        if epoch % display_step == 0:
            feed_train = {x: trainimg[1: 100], y: testlabel[1: 100]}
            feedt_test = {x: mnist.test.images, y: mnist.test.labels}
            train_acc = sess.run(accr, feed_dict=feed_train)
            test_acc = sess.run(accr, feed_dict=feedt_test)

            print("Eppoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f" %
                  (epoch, training_epochs, avg_cost, train_acc, test_acc))
    saver.save(sess, "./MNIST_model/MNIST.model")

print("Done.")


