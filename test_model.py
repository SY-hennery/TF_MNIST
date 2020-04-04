import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import mnist_func as m_func

mnist = input_data.read_data_sets("./MNIST_data/",one_hot = True)
sess = tf.Session()

saver = tf.train.import_meta_graph('./MNIST_model/MNIST.model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./MNIST_model'))

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
func = graph.get_tensor_by_name("func:0")

test_pic,test_label = mnist.train.next_batch(1)
ans = sess.run(func,feed_dict={x:test_pic})

print('predict:',m_func.parse_lables(ans))
print('real:',m_func.parse_lables(test_label))

m_func.parse_image(test_pic)
print('Done')