
# coding: utf-8

# In[1]:


from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[3]:


mnist.train.images


# In[4]:


import tensorflow as tf


# In[5]:


mnist.train.labels


# In[6]:


x = tf.placeholder(tf.float32, [None, 784])


# In[7]:


W = tf.Variable(tf.zeros([784, 10]))


# In[8]:


b = tf.Variable(tf.zeros([10]))


# In[9]:


y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[10]:


y_ = tf.placeholder(tf.float32, [None, 10])


# In[11]:


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[12]:


train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)


# In[13]:


sess = tf.InteractiveSession()


# In[14]:


tf.global_variables_initializer().run()


# In[15]:


for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# In[16]:


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


# In[18]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[20]:


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[ ]:




