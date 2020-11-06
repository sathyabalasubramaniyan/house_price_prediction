import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso,ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
# read the csv file 
Data = pd.read_csv(r'C:\Users\hp\Desktop\Sathya\ml\kc_house_data.csv')
print("*********************************DATASET*******************************************") 
print(Data.head())
print("*********************************INFORMATION*******************************************") 
print(Data.info())
print("*********************************ROWS AND COLUMNS******************************************") 
print(Data.shape)
#print("**************************** Columns having null values *****************************")
#print(list(Data.columns.values.tolist()) )
#for col in list(Data.columns.values.tolist()):
# Data[col]=Data[col].fillna('None')
if True  in  Data.isnull().any():
    print('There is null value in the dataset')
else:
  print('There is no null value in the dataset')
print("**************************** DESCRIBE*****************************")
print(Data.describe().transpose().to_string())
Data = Data.drop('id',axis=1)
Data = Data.drop('zipcode',axis=1)
Data['date'] = pd.to_datetime(Data['date'])
Data['month'] = Data['date'].apply(lambda date:date.month)
Data['year'] = Data['date'].apply(lambda date:date.year)
# check correlation
print("**************************** CORRELATION*****************************")
print(Data.corr()['price'].sort_values(ascending=False))
print("**************************** CORRELATION BETWEEN FEATURES*****************************")  
correlations=Data.corr()
attrs = correlations.iloc[:-1,:-1] # all except target

threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
    .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), 
        columns=['Attribute Pair', 'Correlation'])

    # sorted by absolute value
unique_important_corrs = unique_important_corrs.iloc[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

print(unique_important_corrs)
Data = Data.drop('date',axis=1)
X = Data.drop('price',axis =1).values
y = Data['price'].values
print("*********************************FEATURES*******************************************") 
print(Data.drop('price',axis =1))
print("*********************************TARGET*******************************************") 
print(Data['price'])
#splitting Train and Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
#print(X_train, X_test)
#standardization scaler - fit&transform on train, fit only on test
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))
#print(X_train, X_test)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
y_predd = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predd})
df1 = df.head(10)
print("****************************************************************************") 
print(' MULTIPLE LINEAR REGRESSION')
print(df1)
rr = Ridge(alpha=0.01,normalize=True)
rr.fit(X_train, y_train)
pred_train_rr= rr.predict(X_train)
pred_test_rr= rr.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': pred_test_rr})
df1 = df.head(10)
print("****************************************************************************") 
print('RIDGE REGRESSION')
print(df1)
model_lasso = Lasso(alpha=10)
model_lasso.fit(X_train, y_train) 
pred_train_lasso= model_lasso.predict(X_train)
pred_test_lasso= model_lasso.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': pred_test_lasso})
df1 = df.head(10)
print("****************************************************************************") 
print('LASSO REGRESSION')
print(df1)
model_enet = ElasticNet(alpha = 0.01)
model_enet.fit(X_train, y_train) 
pred_train_enet= model_enet.predict(X_train)
pred_test_enet= model_enet.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': pred_test_enet})
df1 = df.head(10)
print("****************************************************************************") 
print('ELASTICNET REGRESSION')
print(df1)
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,learning_rate = 0.1, loss = 'ls')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(10)
print("****************************************************************************") 
print('GRADIENTBOOST REGRESSION')
print(df1)
x=[regressor.score(X_train,y_train)*100,rr.score(X_train,y_train)*100,model_lasso.score(X_train,y_train)*100,model_enet.score(X_train,y_train)*100,clf.score(X_train,y_train)*100]
y=[regressor.score(X_test, y_test)*100,rr.score(X_test, y_test)*100,model_lasso.score(X_test, y_test)*100,model_enet.score(X_test, y_test)*100,clf.score(X_test, y_test)*100]
z=[metrics.mean_absolute_error(y_test, y_predd),metrics.mean_absolute_error(y_test,pred_test_rr),metrics.mean_absolute_error(y_test,pred_test_lasso),metrics.mean_absolute_error(y_test,pred_test_enet),metrics.mean_absolute_error(y_test, y_pred)]
h=[metrics.mean_squared_error(y_test, y_predd),metrics.mean_squared_error(y_test,pred_test_rr),metrics.mean_squared_error(y_test,pred_test_lasso),metrics.mean_squared_error(y_test,pred_test_enet),metrics.mean_squared_error(y_test, y_pred)]
g=[np.sqrt(metrics.mean_squared_error(y_test, y_predd)),np.sqrt(metrics.mean_squared_error(y_test,pred_test_rr)), np.sqrt(metrics.mean_squared_error(y_test,pred_test_lasso)),np.sqrt(metrics.mean_squared_error(y_test,pred_test_enet)),np.sqrt(metrics.mean_squared_error(y_test, y_pred))]
v=[metrics.explained_variance_score(y_test,y_predd),metrics.explained_variance_score(y_test,pred_test_rr),metrics.explained_variance_score(y_test,pred_test_lasso),metrics.explained_variance_score(y_test,pred_test_enet),metrics.explained_variance_score(y_test, y_pred)]
print("****************************************************************************") 
data = pd.DataFrame(np.column_stack([x,y,z,h,g,v]),columns=['Train Score','Test Score','Mean Absolute Error','Mean Squared Error','Root Mean Squared Error','Variance'],index= ['Linear Regression Model:','Ridge Regression Model:','Lasso Regression Model:','ElasticNet Regression Model:','GradientBoosting Regression Model:'])
print(data.to_string())
print("****************************************************************************")
fig = plt.figure(figsize=(10,5))
fig.add_subplot(3,2,1)
plt.scatter(y_test,y_predd)
plt.title(" MULTIPLE LINEAR REGRESSION ")
plt.ylabel('predicted value')
plt.xlabel('Actual price')
fig.add_subplot(3,2,2)
plt.scatter(y_test,pred_test_rr,color='purple')
plt.title("RIDGE REGRESSION ")
plt.ylabel('predicted value')
plt.xlabel('Actual price');
fig.add_subplot(3,2,3)
plt.scatter(y_test,pred_test_lasso,color='red')
plt.title("LASSO REGRESSION ")
plt.ylabel('predicted value')
plt.xlabel('Actual price');           
fig.add_subplot(3,2,4)
plt.scatter(y_test,pred_test_enet,color='gold')
plt.title("ELASTICNET REGRESSION ")
plt.ylabel('predicted value')
plt.xlabel('Actual price');
fig.add_subplot(3,2,5)
plt.scatter(y_test,y_pred,color='green')
plt.title("GRADIENTBOOST REGRESSION' ")
plt.ylabel('predicted value')
plt.xlabel('Actual price');
plt.tight_layout()
fig.tight_layout(pad=1.5,w_pad=1.5,h_pad=10.0)
plt.show()























