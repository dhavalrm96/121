#!/usr/bin/env python
# coding: utf-8

# In[242]:


## Load library 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.simplefilter('ignore')


# # Load the Dataset

# In[243]:


data = pd.read_csv('Bengaluru_House_Data.csv')


# In[244]:


data.head()


# In[245]:


data.shape


# In[246]:


data['area_type'].unique()


# In[247]:


data['area_type'].value_counts()


# Drop the features that are not required to build our model

# In[248]:


data1= data.drop(['area_type','society', 'balcony', 'availability'], axis= 'columns')


# In[249]:


data1


# # Data Cleaning: Handle the NA values

# In[250]:


data1.isnull().sum()


# In[251]:


data2 = data1.dropna()
data2.isnull().sum()


# In[252]:


data2.dtypes


# In[253]:


data2['size'].unique()


# # Feature Engineering

# Add new feature for bhk

# In[254]:


data2['Bhk'] = data2['size'].apply(lambda x : int(x.split(' ')[0]))


# In[255]:


data2.head()


# In[256]:


data2['Bhk'].unique()


# In[257]:


data2[data2.Bhk >20]


# # Insight

# There is some problem in the total_Sqft column it shows 43 bedroom in 2400 sq ft aewa..

# In[258]:


data2['total_sqft'].unique()


# Their is some range values in the 'total_sqft' column as like "1133 - 1384" let's convert it into a single value..

# In[259]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[260]:


data2[~data2['total_sqft'].apply(is_float)].head(20)


# Above shows that total_sqft can be a range (e.g. 2100-2850). For such case we can just take average of min and max value in the range. There are other cases such as 34.46Sq. Meter which one can convert to square ft using unit conversion. I am going to just drop such corner cases to keep things simple

# In[261]:


def convert_sqft_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return(float(tokens[0]) + float(tokens[1]))/2
    try: 
        return float(x)
    except:
        return None


# In[262]:


data2['total_sqft'] = data2['total_sqft'].apply(convert_sqft_num)


# In[263]:


data2.head()


# In[264]:


# Creating a new column to find the price per sq-ft

data2['Price_per_sqft'] = data2['price']*100000/data2['total_sqft']
data2.head()


# # Dimensionality Reduction

# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later on when do one hot encoding, it will help us with having fewer dummy columns

# In[265]:


data2['location'].unique()


# In[266]:


len(data2['location'])


# In[267]:


## let's check each location in column

data2['location'] = data2['location'].apply(lambda x : x.strip())

locations = data2.groupby('location')['location'].agg('count').sort_values(ascending= False)

locations


# In[268]:


## let's check the location is 1 to 10 times


locations[locations<=10]


# In[269]:


len(locations[locations<=10])


# In[270]:


locations_less_than_10 = locations[locations<=10]
locations_less_than_10


# In[271]:


##let's put the "locations_less_than_10" in name of "Other" category..

data2['location'] = data2['location'].apply(lambda x : 'other' if x in locations_less_than_10 else x)


# In[272]:


data2.head(10)


# In[273]:


## let's check the sq ft area's  as per 1 bedroom consist 300 sqft..

data2 [data2['total_sqft']/data2['Bhk']<300].head()


# # Insights

# Check above data points. We have 6 bhk apartment with 1020 sqft. Another one is 8 bhk and total sqft is 600. These are clear data errors that can be removed safely

# In[274]:


data2.shape


# In[275]:


## remove the outliers 

data2 = data2[~(data2['total_sqft']/data2['Bhk']<300)]


# In[276]:


data2.shape


# In[277]:


## let's check the price per sqft column

data2.Price_per_sqft.describe()


# There are some extreme values. So, we have to remove them or consize them..

# In[278]:


## let's remove outliers in the price_per_sqft

def remove_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.Price_per_sqft)
        st = np.std(subdf.Price_per_sqft)
        reduced_df = subdf[(subdf.Price_per_sqft>(m-st)) & (subdf.Price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index= True)
    return df_out

house = remove_outliers(data2)


# In[279]:


house.shape


# # Visualizations

# In[280]:


##check the outliers for the BHK column

import matplotlib
matplotlib.rcParams['figure.figsize']=(15,8)

plt.hist(house.Price_per_sqft, rwidth=0.8)
plt.xlabel('Price per Sqft')
plt.ylabel('Count')


# The price is between the range of 1000-10000 price sqft.

# In[281]:


house['bath'].unique()


# In[282]:


house[house['bath']>10]


# In[283]:


plt.hist(house.bath, rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[284]:


house[house.bath>house.Bhk+2]


# if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed

# In[285]:


data4 = house[house.bath<house.Bhk+2]

data4


# In[286]:


## let's drop some columns for the model building 

data5 = data4.drop(['size', 'Price_per_sqft'], axis = 'columns')

data5.head()


# # Use One hot encoding For location

# In[287]:


dummies = pd.get_dummies(data5.location)

dummies


# In[288]:


data6 = pd.concat([data5,dummies.drop('other' , axis= 'columns')], axis = 'columns')


# In[289]:


data6.head()


# In[290]:


final = data6.drop('location', axis = 'columns')

final.head()


# # Build a Model

# In[291]:


final.shape


# In[292]:


X = final.drop(['price'], axis = 'columns')
y = final.price


# In[293]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[294]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=10)


# In[295]:


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test,y_test)


# # Use K-Fold cross validation to measure accuracy of our LinearRegression Model

# In[296]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# # Find best model using GridSearch CV

# In[297]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


# In[307]:


def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# Based on above results we can say that LinearRegression gives the best score. Hence we will use that.

# In[319]:


def predict_price(location, sqft,bath, bhk):
    loc_index=np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0 :
      x[loc_index] = 1

    return lr.predict([x])[0]


# In[320]:


predict_price('1st Phase JP Nagar', 1000, 2, 2)


# In[321]:


predict_price('Indira Nagar',1000,2,2)


# # Export the Tested Model to a pickel file

# In[ ]:


import pickle
picke.dump()
## with open('banglore_home_prices_model.pickle','wb') as f:
    ## pickle.dump(lr,f)
        


# # Export location and column information to a file that will be useful late on in our prediction application

# In[322]:


import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("Columns.json","w") as f:
  f.write(json.dumps(columns))

