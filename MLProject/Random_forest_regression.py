import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
np.random.seed(0)

def normalize_df(X):
    l= list(X.select_dtypes(include='float64'))
    norm_df=X.copy()
    for col in l:
        norm_df[col]=(X[col]-X[col].mean())/X[col].std()
    return norm_df

#Load car data
data = pd.read_csv('CarPriceEnhanced.csv')
CarData= data.copy()
X=CarData.iloc[:,1:26] #Features

Y=CarData['price'] #Label
norm_X=normalize_df(X)
norm_Y=(Y-Y.mean())/Y.std()
Y_STD=Y.std()
Y_MEAN=Y.mean()
#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(norm_X, norm_Y, test_size = 0.30,shuffle=True)
#Get the correlation between the features
corr = CarData.corr()
#Top 30% Correlation training features with the Value
top_feature = corr.index[abs(corr['price']>0.3)]
top_feature=[col for col in top_feature if col != 'price']

plt.subplots(figsize=(12, 8))
top_corr = CarData[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


cls =RandomForestRegressor(random_state=1)
cls.fit(X_train[top_feature],y_train)
prediction= cls.predict(X_test[top_feature])
score= r2_score(y_test,prediction)
pickle.dump(cls,open('RFR.pkl','wb'))
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))

true_car_value=(np.asarray(y_test)[0]*Y_STD)+Y_MEAN
predicted_car_value=(prediction[0]*Y_STD)+Y_MEAN
print(score)
print('True value for the first player in the test set in millions is : ' + str(true_car_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_car_value))
print()