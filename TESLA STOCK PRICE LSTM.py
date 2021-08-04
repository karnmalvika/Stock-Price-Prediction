#!/usr/bin/env python
# coding: utf-8

# # Final project :Stock Price predictor¶

# Predicting how the stock market will perform is one of the most difficult things to do. There are so many factors involved in the prediction – physical factors vs. psychological, rational and irrational behaviour, etc. 
# 
# All these aspects combine to make share prices volatile and very difficult to predict with a high degree of accuracy. Here in this project we will use machine learning as a game changer in this domain Using features like the latest announcements about an organization, their quarterly revenue results, etc., machine learning techniques have the potential to unearth patterns and insights we didn’t see before, and these can be used to make unerringly accurate
# 
# In this project we will work with historical data about the stock prices of a publicly listed stock price of this company like Microsoft ,Apple, Tesla
# 
# 

# We’ll kick of by importing NumPy for scientific computation, Matplotlib for plotting graphs, and Pandas to aide in loading and manipulating our datasets.

# In[1]:


#import packages
import pandas as pd
import numpy as np


import seaborn as sns

#to plot within notebook
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#import statsmodels.api as sm


# In[2]:


#Read the dataset
df=pd.read_csv('Stock_data.csv')


# We check the head of our dataset to give us a glimpse into the kind of dataset we’re working with.
# 

# In[3]:


df.head()


# There are multiple variables in the dataset – date, open, high, low, last, close, total_trade_quantity, and turnover.
# 
# The columns Open and Close represent the starting and final price at which the stock is traded on a particular day.
# 
# High, Low and Last represent the maximum, minimum, and last price of the share for the day.
# 
# Total Trade Quantity is the number of shares bought or sold in the day 
# 
# Another important thing to note is that the market is closed on weekends and public holidays.Notice the above table again, some date values are missing – 1984-09-08,1984-09-09, 8th and 9th fall on a weekend.

# In[4]:


df.shape


# By using shape function we can find number of rows and columns
# 
# For this dataset we have 506 rows with 10 columns
# 
# Let us see the data types of each of the columns to get an idea of the type of data we have

# In[5]:


df.info()


# 'Date' column should be date-type we will convert it to date type using to_datetime() function

# In[6]:


#Check for missing values
df.isnull().sum()


# Looks like we have cleaned dataset means its good to go ahead

# In[7]:


df['Stock'].unique()


# The master dataset contain the data for various companies like Apple,Tesla,Facebook,Microsoft
# Out of this we will first choose TESLA dataset for our analysis then we can do similar process for remaining companies also.

# In[8]:


ts_df=df.loc[df['Stock']=='MSFT']
ts_df.head()


# The Open column is the starting price while the Close column is the final price of a stock on a particular trading day. The High and Low columns represent the highest and lowest prices for a certain day.
# 
# 

# The profit or loss calculation is usually determined by the closing price of a stock for the day, hence we will consider the closing price as the target variable. Let’s plot the target variable to understand how it’s shaping up in our data:

# # Exploratory Data Analysis
# Summary statistics
# 
# Let us start with getting the summary statistics on the given data

# In[9]:


ts_df.describe()


# Describe() function will give us all necessary summary statistics of each column at one place
# 
# 

# In[10]:


#Setting index as date
ts_df['Date'] = pd.to_datetime(ts_df.Date,format='%Y-%m-%d')
ts_df.index = ts_df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(ts_df['Close'], label='Close Price history')


# # Building LSTM Model

# Long Short Term Memory (LSTM)
# Introduction
# 
# LSTMs are widely used for sequence prediction problems and have proven to be extremely effective. The reason they work so well is because LSTM is able to store past information that is important, and forget the information that is not. LSTM has three gates:
# 
# The input gate: The input gate adds information to the cell state
# 
# The forget gate: It removes the information that is no longer required by the model
# 
# The output gate: Output Gate at LSTM selects the information to be shown as output
# 
# For now, let us implement LSTM as a black box and check it’s performance on our particular data.
# 
# 

# In[11]:


#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# In order to build the LSTM, we need to import a couple of modules from Keras as above:
# 
# Sequential for initializing the neural network
# 
# Dense for adding a densely connected neural network layer
# 
# LSTM for adding the Long Short-Term Memory layer
# 
# Dropout for adding dropout layers that prevent overfitting

# ########################################################################
# 
# We will first sort the dataset in ascending order and then create a separate dataset so that any new feature created does not affect the original data.

# In[12]:


#creating dataframe
data = ts_df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(ts_df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]


# In[13]:


#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)


# In[14]:


new_data.shape


# In[15]:


#creating train and test sets
dataset = new_data.values

train = dataset[0:6386,:]
valid = dataset[6386:,:]


# we have to scale our data for optimal performance. In our case, we’ll use Scikit- Learn’s MinMaxScaler and scale our dataset to numbers between zero and one.
# 
# ###Creating Data with Timesteps
#  
# LSTMs expect our data to be in a specific format, usually a 3D array. We start by creating data in 60 timesteps and converting it into an array using NumPy. Next, we convert the data into a 3D dimension array with X_train samples, 60 timestamps, and one feature at each step.
# 

# In[16]:


#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# We add the LSTM layer and later add a few Dropout layers to prevent overfitting. We add the LSTM layer with the following arguments:
# 
# 50 units which is the dimensionality of the output space
# return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence
# input_shape as the shape of our training set.
# Thereafter, we add the Dense layer that specifies the output of 1 unit. After this, we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error. This will compute the mean of the squared errors. Next, we fit the model to run on 10 epochs with a batch size of 32. 

# In[17]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)


# The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
# 
# One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters. An epoch is comprised of one or more batches. For example, as above, an epoch that has one batch is called the batch gradient descent learning algorithm.
# 
# 

# The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.
# 
# 

# In[18]:


#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


# In[19]:


rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms


# # Plotting the Results
#  
# Finally, we use Matplotlib to visualize the result of the predicted stock price and the real stock price.

# In[25]:


#for plotting
plt.figure(figsize=(20,15))
plt.rcParams.update({'font.size': 22})
train = new_data[:6386]
valid = new_data[6386:]
valid['Predictions'] = closing_price

valid.index = new_data[6386:].index
train.index = new_data[:6386].index

plt.plot(train['Close'],label = 'Tesla Stock Price')
plt.plot(valid[['Close','Predictions']],label = 'Predicted Tesla Stock Price')
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.legend()
plt.show()


# From the plot we can see that the real stock price went up while our model also predicted that the price of the stock will go up. This clearly shows how powerful LSTMs are for analyzing time series and sequential data.

# In[23]:


valid


# From above output we can say that how well our model can predict for particular date .

# # Conclusion
#  
# There are a couple of other techniques of predicting stock prices such as moving averages, linear regression, K-Nearest Neighbours, ARIMA and Prophet. These are techniques that one can test on their own and compare their performance with the Keras LSTM.

# The LSTM model can be tuned for various parameters such as changing the number of LSTM layers, adding dropout value or increasing the number of epochs. But are the predictions from LSTM enough to identify whether the stock price will increase or decrease? Certainly not!
# 
# As stock price is affected by the news about the company and other factors like demonetization or merger/demerger of the companies. There are certain intangible factors as well which can often be impossible to predict beforehand.
# 
# 

# Similarly we can do this prediction for remaining companies of dataset i.e. facebook, tesla,microsoft
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# 
