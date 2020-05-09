#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf 
import random
import numpy as np
from sklearn import datasets
tf.compat.v1.disable_eager_execution()


# In[2]:


iris = datasets.load_iris()


# In[4]:


X = iris.data
y = iris.target


# In[5]:


a = list(range(len(X)))
random.shuffle(a)


# In[19]:


X_train = []
y_train = []
X_test = []
y_test = []


# In[20]:


partition = int(0.8*len(a))


# In[21]:


train_indices = a[:partition]
test_indices = a[partition :]


# In[22]:


for i in train_indices:
    X_train.append(X[i])
    val = [0,0,0]
    val[y[i]] =1
    y_train.append(val)


# In[23]:


for i in test_indices:
    X_test.append(X[i])
    val = [0,0,0]
    val[y[i]] = 1
    y_test.append(val)
    


# In[25]:


n = len(X_train[0])
k = len(y_train[0])


# In[27]:


learning_rate = 0.005
batch_size = 50
n_hidden_l = 4


# In[28]:


training_epochs = 3000
display_step = 1
stddev = 0.01 


# In[30]:


# Network Parameters
n_input = len(X_train[0])
n_classes = len(y_train[0])
logs_path = 'C:/tmp/tf_logs/'

# In[34]:


#Graph Inputs
input_data = tf.compat.v1.placeholder(float, name='input_data',shape=[None,n_input])
output_data = tf.compat.v1.placeholder(float,name='output_data', shape=[None,n_classes])


# In[52]:


# Constructing Model 
def multilayer_perceptron(x,weights,biases):
    #Hidden Layer with ReLu activation fn
    layer_1 = tf.matmul(x,weights['h1'])+ biases['b1']
    layer_1 = tf.nn.relu(layer_1)
    #Output layer with linear activation fn
    out_layer = tf.matmul(layer_1,weights['out']) + biases['out']
    return out_layer 


# In[53]:


# Declaring weight and biases variable
weights = {
    'h1': tf.compat.v1.Variable(tf.random.normal([n_input, n_hidden_l], stddev=stddev)),
    'out': tf.compat.v1.Variable(tf.random.normal([n_input,n_classes], stddev=stddev))
}
biases = {
    'b1': tf.compat.v1.Variable(tf.random.normal([n_hidden_l],stddev =stddev)),
    'out': tf.compat.v1.Variable(tf.random.normal([n_classes],stddev =stddev))
}


# In[ ]:




# In[54]:


pred = multilayer_perceptron(input_data, weights, biases)

#Cost and Loss 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits = pred ,labels = output_data))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
tf.summary.scalar('cost', cost)


merged = tf.compat.v1.summary.merge_all()
summary_writer = tf.compat.v1.summary.FileWriter(logs_path, sess.graph)

# In[ ]:
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

final_weights = None
final_biases = None

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(X_train)/batch_size)
    #Loop for all baches
    for i in range(total_batch):
        l = i*batch_size
        h = (i+1)*batch_size
        
        batch_x = X_train[i*batch_size : (i+1)* batch_size]
        batch_y = y_train[i*batch_size : (i+1)* batch_size]
        #Run cost fun to get optimize values of Weights  
        _,c,final_weights,final_biases = sess.run([optimizer,cost,weights,biases],
                                                  feed_dict = {input_data : batch_x,
                                                               output_data : batch_y
                                                               })
        #summary_writer.add_summary(summary, epoch * total_batch + i)
        #Compute avg loss
        avg_cost += c/total_batch
        
        if epoch% display_step ==0 :
            print("Epoch:",'%04d'% (epoch +1), "cost=",
                      "{:9f}".format(avg_cost))


# In[ ]:
correct = 0
weights['h1'] = tf.cast(weights['h1'],tf.float64)
weights['out'] = tf.cast(weights['out'], tf.float64)

predictions = sess.run(pred, feed_dict = {
        input_data : X_test,
        output_data : y_test
        })
correct = 0
for i in range(len(predictions)):
    if np.argmax(predictions[i]) == np.argmax(y_test[i]):
        correct += 1
acc = 100.0* correct/ len(predictions)
print('Accuracy', 100.0* correct/ len(predictions))
tf.summary.scalar('Accuracy', acc)


