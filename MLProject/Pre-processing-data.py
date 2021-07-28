import numpy as np
import pandas as pd


df=pd.read_csv('CarPrice_training.csv')

df_clean=df.copy()

df_clean['wheelbase'].fillna(method='bfill',inplace=True)
for i in range(9,12):
    df_clean.iloc[:,i].fillna(method='ffill',inplace=True)

for i in range(12,22):
    df_clean.iloc[:,i].fillna(method='bfill',inplace=True)

df_clean['citympg'].fillna(df['citympg'].mean(),inplace=True)

data=df_clean['fueltype'] 

data=pd.get_dummies(data)

df_clean=df_clean.drop('fueltype',axis=1)
df_clean=df_clean.join(data)

data1=df_clean['enginelocation']
data1=pd.get_dummies(data1)
df_clean=df_clean.drop('enginelocation',axis=1)
df_clean=df_clean.join(data1)


data2=df_clean['aspiration']
data2=pd.get_dummies(data2)
df_clean=df_clean.drop('aspiration',axis=1)
df_clean=df_clean.join(data2)


df_clean['doornumber']=df_clean['doornumber'].map({'four': 4,'two':2})

df_clean['cylindernumber']=df_clean['cylindernumber'].map({'four': 4,'three':3,'five':5 ,'six': 6,'eight': 8 , 'twelve':12})

l= list(df_clean.select_dtypes(include='object'))
df_l=df_clean.copy()

for col in l:
    df_l[col] = df_l[col].astype('category')
    df_l[col] = df_l[col].cat.codes

df_2 =df_clean['price'].copy()
df_2.head()

df_l=df_l.join(df_2)





df_l.to_csv('CarPriceEnhanced.csv')