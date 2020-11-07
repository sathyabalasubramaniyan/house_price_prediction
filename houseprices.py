
# import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
Data = pd.read_csv(r'C:\Users\hp\Desktop\Sathya\ml\kc_house_data.csv')
Data = Data.drop(['id','zipcode','date'],axis=1)
X = Data.drop('price',axis =1).values
y = Data['price'].values
X_train, X_tes, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
d = {}
list=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']
print('PREDICTION FOR NEW HOUSE')
print("NOTE: If you don't know any values,please enter 0 for those values")
for i, f in zip(range(17),list):
    print('*'*40)
    keys = f # here i have taken keys as strings
    values = float( input('enter value for'+' '+f)) # here i have taken values as integers
    d[keys] = values
print(d)
df = pd.DataFrame(d, columns = list, index=['value'])
xtest=df.values
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(xtest.astype(np.float))
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,learning_rate = 0.1, loss = 'ls')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)
print('THE PREDICTION PRICES FOR HOUSE IS AROUND' ,y_pred[0])
