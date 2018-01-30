# import mnist dataset
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x  = tf.placeholder(tf.float32, [None, 784], name='x')
x_image = tf.reshape(x, [-1, 28, 28, 1])

y_ = tf.placeholder(tf.float32, [None, 10],  name='y_')

# define convolutional layers

# Convolutional layer 1
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Convolutional layer 2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

h_conv2 = tf.nn.relu( tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')  + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1 = tf.Variable(tf.truncated_normal(([7 * 7 * 64, 1024]), stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# keep_prob  = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = tf.Variable(tf.truncated_normal(([1024, 10]), stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name='y')

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  max_steps = 10000
  for step in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if (step % 100) == 0:
      print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))