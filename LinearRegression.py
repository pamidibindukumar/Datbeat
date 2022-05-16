#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import re
import pandas as pd
#https://www.askpython.com/python/examples/logistic-regression-from-scratch#:~:text=Implementing%20Logistic%20Regression%20from%20Scratch%20Step-1%3A%20Understanding%20the,we%20want%20to%20optimize%20a%20loss...%20Step-3%3A%20 
class LinearRegression:
    def __init__(self,x,y):      
        self.intercept = np.ones((x.shape[0], 1))  
        self.x = np.concatenate((self.intercept, x), axis=1)
        self.weight = np.random.normal(self.x.shape[1])
        self.y = y
    #method to calculate the Loss
    def loss(self, y_pred, y):
        '''
        loss function for logistic regression
        '''
        return np.square(y-y_pred).mean()

    #Method for calculating the gradients
    def gradient_descent(self, X, y_pred, y):
        '''
        Gradient descent
        '''
        return np.dot(X.T,(y-y_pred))/y.shape[0]
#         return np.dot(X.T, (h - y)) / y.shape[0]
 
    def predicted_value(x,w):
        return np.dot(x,w)+self.intercept
    
    def fit(self, lr , iterations):
        for i in range(iterations):
            y_pred = self.predicted_value(self.x, self.weight)
             
            loss = self.loss(y_pred,self.y)
 
            dW = self.gradient_descent(self.x , y_pred, self.y)
             
            #Updating the weights
            self.weight -= lr * dW
 
        return print('fitted successfully to data')
     
    #Method to predict the class label.
    def predict(self, x_new ):
        x_new = np.concatenate((self.intercept, x_new), axis=1)
        result = self.predicted_value(x_new, self.weight)
#         result = result >= treshold
        y_pred = result
        for i in range(len(y_pred)):
            if result[i] == True: 
                y_pred[i] = 1
            else:
                continue
                 
        return y_pred


# In[ ]:





# In[4]:


import pandas as pd
data_red=pd.read_csv(r'C:\Assignment\winequality\winequality\winequality-red.csv')
data_white=pd.read_csv(r'C:\Assignment\winequality\winequality\winequality-white.csv')
data=pd.concat([data_red,data_white])
data.shape


# In[10]:


import re
columns=data.columns[0].split(';')
new_columns=[]
for i in columns:
    i=re.sub('"','',i)
    new_columns.append(i)
new_columns


# In[32]:


# data.apply(lambda x:x.split(';'))
# type(data)
data.columns=['value']
data.head()
series=data.value.str.split(';')
type(series)


# In[39]:


np_data=np.array(np_data.values.tolist())
print(np_data.shape)


# In[26]:


import numpy as np
np_data=np.array(np_data)
np_data.shape

