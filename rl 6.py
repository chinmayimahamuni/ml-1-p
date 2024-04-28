#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym


# In[2]:


pip install gym


# In[3]:


import gym
import numpy as np
import tensorflow as tf


# In[4]:


tensor = tf.constant([[1, 2], [3, 4]])


# In[5]:


print("Tensor:")


# In[6]:


print(tensor)


# In[11]:


variable = tf.Variable(tf.zeros([2, 2]))
print("\nVariable:")
print(variable)

variable.assign_add(tf.ones([2, 2]))

print("\nUpdated Variable:")
print(variable)

env = gym.make('CartPole-v1')

state = env.reset()


# In[13]:


for _ in range(100):
    
    env.render()
    action = env.action_space.sample()
    
    # perform the action in the environment
    result = env.step(action)
    
    #unpack the first four values
    next_state, reward, done, info = result[:4]
    
    #If episode is finished , reset the environment
    if done:
        state = env.reset()
        
# close the envionment
env.close()


# In[14]:


import gym
import numpy as np
import tensorflow as tf
tensor = tf.constant([[1, 2], [3, 4]])
print("Tensor:")
print(tensor)
variable = tf.Variable(tf.zeros([2, 2]))
print("\nVariable:")
print(variable)

variable.assign_add(tf.ones([2, 2]))

print("\nUpdated Variable:")
print(variable)

env = gym.make('CartPole-v1')

state = env.reset()
for _ in range(100):
    
    env.render()
    action = env.action_space.sample()
    
    # perform the action in the environment
    result = env.step(action)
    
    #unpack the first four values
    next_state, reward, done, info = result[:4]
    
    #If episode is finished , reset the environment
    if done:
        state = env.reset()
        
# close the envionment
env.close()


# In[ ]:




