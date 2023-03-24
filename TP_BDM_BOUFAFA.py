#!/usr/bin/env python
# coding: utf-8

# In[323]:

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.metrics import r2_score

from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential

#check all the files in the input dataset
print(os.listdir("./"))


# In[324]:


fpath='AEP_HOURLY.csv'

df=pd.read_csv(fpath)
df.head()


# In[325]:


df = pd.read_csv(fpath, index_col='Datetime', parse_dates=['Datetime'])
df.head()


# In[326]:


df.isna().sum()


# In[327]:


df.plot(figsize=(16,4),legend=True)

plt.title('DOM hourly power consumption data - avant la normalisation')

plt.show()


# In[328]:


scaler = sklearn.preprocessing.MinMaxScaler()
df['AEP_MW']=scaler.fit_transform(df['AEP_MW'].values.reshape(-1,1))
df_norm = df
df_norm


# In[329]:


df_norm.plot(figsize=(16,4),legend=True)

plt.title('dataset - Après normalisation')

plt.show()


# In[330]:


df_norm.shape


# In[331]:


data_training = df[0:100000].copy()
data_training


# In[332]:


data_test = df[100000:121273].copy()
data_test


# In[333]:


data_training=np.array(data_training)


# In[334]:


X_train = []
y_train = []
for i in range(5, data_training.shape[0]):
    X_train.append(data_training[i-5:i])
    y_train.append(data_training[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[335]:


X_train.shape, y_train.shape


# In[336]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
model = Sequential()
model.add(SimpleRNN(units = 50, activation = 'relu',return_sequences = True,  input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.1))

model.add(SimpleRNN(units = 50, activation = 'relu', return_sequences = True))
model.add(Dropout(0.1))

model.add(SimpleRNN(units = 60, activation = 'relu'))
model.add(Dropout(0.1))


model.add(Dense(units =1))


# In[337]:


model.summary()


# In[338]:


model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs=8, batch_size=32)


# In[339]:


past_data = df.tail(5)
data = past_data.append(data_test, ignore_index = True)
data.head()


# In[340]:


inputsTest = scaler.transform(df)
inputsTest


# In[341]:


inputsTest.shape[0]


# In[342]:


X_test = []
y_test = []

for i in range(5, inputsTest.shape[0]):
    X_test.append(inputsTest[i-5:i])
    y_test.append(inputsTest[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape

# In[343]:


y_pred = model.predict(X_test)
y_pred

# In[344]:


y_pred = y_pred
y_test = y_test
for i in range(0,8):
    if y_pred[i]>0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0

# In[345]:


plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'indicateur réel')
plt.plot(y_pred, color = 'blue', label = 'indicateur prédit')
plt.title("Prédiction de la consommation d'energie dans le temps")
plt.xlabel('Time')
plt.ylabel('Normalized power consumption scale')
plt.legend()
plt.show()





# In[ ]:




