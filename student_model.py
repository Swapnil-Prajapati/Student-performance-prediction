#!/usr/bin/env python
# coding: utf-8

# In[2]:


# we must import the libraries once again since we haven't imported them in this file
import numpy as np
import tensorflow as tf


# In[4]:


npz = np.load('Student-mat_data_train.npz')
train_inputs = npz['inputs'].astype(np.float)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = npz['targets'].astype(np.int)

# we load the validation data in the temporary variable
npz = np.load('Student-mat_data_validation.npz')
# we can load the inputs and the targets in the same line
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

# we load the test data in the temporary variable
npz = np.load('Student-mat_data_test.npz')
# we create 2 variables that will contain the test inputs and the test targets
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)


# In[17]:


input_size = 16
output_size = 2
hidden_layer_size = 50

model = tf.keras.Sequential([
    
                                
                                tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                                tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                                tf.keras.layers.Dense(output_size, activation = 'sigmoid')
    
                           ])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

batch_size = 32
max_epochs = 50

early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)

model.fit(train_inputs, train_targets, batch_size = batch_size, epochs = max_epochs, callbacks = [early_stopping],
                                  validation_data = (validation_inputs, validation_targets),
                                     verbose=2)


# In[18]:


test_loss, test_accuracy= model.evaluate(test_inputs, test_targets)


# In[ ]:




