#!/usr/bin/env python
# coding: utf-8

# # Vehicle Price Prediction With Neural networks

# Our data is taken from "https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes" on the Kaggle platform. Our dataset consists of 100,000 used car information used in the UK. The dataset consists of different car models. In the study, Mercedes brand automobile data were discussed.

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import warnings as w
w.filterwarnings('ignore')


# We load the dataset from excel.

# In[6]:


dataFrame= pd.read_excel("merc.xlsx")


# In[7]:


dataFrame.head()


# The data set consists of 13119 rows of data and 6 different variables. Our variables are year (year), car sales price (price), car's gear type (transmission), tax amount (tax) and Engine Size (EngineSize).

# In[8]:


dataFrame


# ## Make Sense of Data

# In[9]:


dataFrame.describe()


# We check for missing data in the Dataset and none of the variables appear to have missing data.

# In[10]:


dataFrame.isnull().sum()


# ### Graphical Inferences

# Charts provide important clues in our data approach. We examine the graph of the variable of car prices that we are going to predict. When vehicle prices are examined, there are residual data that may cause deviations in the range of 75,000-150,000 TL. These values may cause deviations in our estimates..

# In[13]:


sbn.distplot(dataFrame["price"])


# Looking at the distribution of automobiles by years, it is seen that most of the automobile data belong to 2019.

# In[16]:


plt.figure(figsize=(13,5))

sbn.countplot(dataFrame["year"])


# ### Relationships Between Variables

# The relations of the variables with each other are given.

# In[20]:


dataFrame.corr()


# When we examined the relationship between automobile prices, which is our dependent variable, and other variables in our model, it was seen that there was a positive correlation with the year and engine power. In other words, the prices of cars increase as the year they are produced and the engine power increases. Another noteworthy inference was the tax rate. The higher the tax rate, the higher the price. As car mileage and fuel consumption increase, car prices decrease.

# In[21]:


dataFrame.corr()["price"].sort_values()


# In[22]:


sbn.scatterplot(x="mileage",y="price",data=dataFrame)


# In[23]:


dataFrame.sort_values("price",ascending= False).head(20)


# In[24]:


dataFrame.sort_values("price",ascending= True).head(20)


# In[25]:


len(dataFrame)


# In[27]:


len(dataFrame)*0.01


# ## Data Cleaning

# We perform a 1% subtraction from the data set.

# In[18]:


yuzdeDoksanDokuzDataFrame=dataFrame.sort_values("price",ascending= False).iloc[131:]


# It is seen below that when the 1% residual of the data set is removed, the data set approaches the normal distribution.

# In[32]:


plt.figure(figsize=(7,5))
sbn.displot(yuzdeDoksanDokuzDataFrame["price"])


# In our data set, there is a difference in the average of the data for 1970 compared to other years. The reason for this is that a few 1970 model cars show an imbalance in price-average age matching due to special reasons (antiques, etc.).

# In[33]:


dataFrame.groupby("year").mean()["price"]


# In[35]:


dataFrame=yuzdeDoksanDokuzDataFrame


# The 1970 data were removed from the dataset to bring it to a more accurate point where it could be estimated from the dataset.

# In[36]:


dataFrame = dataFrame[dataFrame.year != 1970]


# ## Model Making

# The categorical variable gear type was removed from the variable's dataset.

# In[21]:


dataFrame= dataFrame.drop("transmission",axis=1)


# The dependent variable (price) of car prices is taken as a series.

# In[22]:


y= dataFrame["price"].values


# Other variables in the data set were taken from the series form as independent variables.

# In[23]:


x= dataFrame.drop("price",axis=1).values


# The activated data set of the Sklearn library was divided into two as training and testing. 70% of the data set was determined as training data and 30% as test data.

# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=10)


# We look at how many rows of data are in the Training and Test dataset.

# In[26]:


len(x_train)


# In[27]:


len(x_test)


# Min-Max normalizasyonu yapmak için sklearn kütüphanesi aktif edildi.

# In[29]:


from sklearn.preprocessing import MinMaxScaler


# In[30]:


scaler = MinMaxScaler()


# In[31]:


x_train = scaler.fit_transform(x_train)


# In[32]:


x_test = scaler.fit_transform(x_test)


# The training model consists of 5 layers, 4 normal and 1 output layer. It is designed by creating 12 neurons in the normal layer and 1 neuron in the output layer. Relu activation function is used in normal layers. In the model, "man" was used as the optimizer and the mean square error was used as the loss.

# In[33]:


from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers  import Dense


# In[34]:


x_train.shape


# In[35]:


model =  Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")


# The model will be trained with the training data. Batch_size 250 is given. Our data set will be given to the model in data groups of 250. Because giving 13,000 data to the model at the same time may cause the model to crash. In model training, iteration (epoch) 250 was given.Test data is given for validation data.

# In[36]:


model.fit(x=x_train, y=y_train,epochs=300,validation_data=(x_test,y_test),batch_size=250)


# ## Evaluation of Results

# Let's examine the losses for training and test data

# In[37]:


kayipVerisi =pd.DataFrame(model.history.history)


# In[38]:


kayipVerisi.head()


# Let's examine the graphs of losses.

# In[39]:


kayipVerisi.plot()


# When we look at the "lost" values of the Training and Test data, it started around 7 in the first iteration and decreased to 1 "loss" in the 25th iteration. From the 25th iteration to the 150th iteration, the "loss" decreased by 0.2. 150th to 300th iteration "losses" of training and test data went in the same direction and there were no overfitting events.

# In[40]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[41]:


tahminDizisi= model.predict(x_test)


# In[42]:


tahminDizisi


# In[43]:


mean_absolute_error(y_test,tahminDizisi)


# When we compared the actual car prices with our estimate for the test data, we found an average difference of 3893 pounds. When we look at the average car prices, we found 24074 liras. According to the user of the data, it can be decided that such an error can be tolerated or further reduced.

# In[44]:


plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-*")


# Conclusion: Our model can be developed according to the user. More data can be collected for this. Test and Training data rates can be changed. The number of iterations can be increased. The number of neurons and layers can be increased.

# In[ ]:




