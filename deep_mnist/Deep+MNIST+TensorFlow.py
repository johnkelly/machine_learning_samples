
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[3]:


x = tf.placeholder(tf.float32, [None, 784])


# In[4]:


y_ = tf.placeholder(tf.float32, [None, 10])


# In[5]:


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


# In[6]:


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# In[7]:


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# In[8]:


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[9]:


W_conv1 = weight_variable([5, 5, 1, 32])


# In[10]:


b_conv1 = bias_variable([32])


# In[11]:


x_image = tf.reshape(x, [-1, 28, 28, 1])


# In[12]:


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)


# In[13]:


h_pool1 = max_pool_2x2(h_conv1)


# In[14]:


# Second Convolution Layer


# In[15]:


W_conv2 = weight_variable([5, 5, 32, 64])


# In[16]:


b_conv2 = bias_variable([64])


# In[17]:


h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)


# In[18]:


h_pool2 = max_pool_2x2(h_conv2)


# In[19]:


# Densely Connected Layer


# In[20]:


W_fc1 = weight_variable([7 * 7 * 64, 1024])


# In[21]:


b_fc1 = bias_variable([1024])


# In[22]:


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])


# In[23]:


h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[24]:


# Dropout


# In[25]:


keep_prob = tf.placeholder(tf.float32)


# In[26]:


h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[27]:


# Read out layer


# In[28]:


W_fc2 = weight_variable([1024, 10])


# In[29]:


b_fc2 = bias_variable([10])


# In[30]:


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# In[31]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))


# In[32]:


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[33]:


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))


# In[ ]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# In[ ]:




