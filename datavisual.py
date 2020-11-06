# import required libraries
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
Data = pd.read_csv(r'C:\Users\hp\Desktop\Sathya\ml\kc_house_data.csv')
Data = Data.drop('id',axis=1)
Data = Data.drop('zipcode',axis=1)
num_feat=Data.columns[Data.dtypes!=object]
num_feat=num_feat[1:-1]
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(Data[col].values,Data.price.values)[0,1])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(10,5))
rects = ax.barh(ind, np.array(values), color='red')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t Price");
plt.show()
correlations=Data.corr()
attrs = correlations.iloc[:-1,:-1] 

threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
    .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), 
        columns=['Attribute Pair', 'Correlation'])
unique_important_corrs = unique_important_corrs.iloc[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]
corrMatrix=Data[list(Data.columns.values.tolist())].corr()
sns.set(font_scale=1.10)
plt.figure(figsize=(10,5))
sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');
plt.show()
fig = plt.figure(figsize=(10,5))
fig.suptitle('salesprices', fontsize=14, fontweight='bold')
fig.add_subplot(3,2,1)
sns.distplot(Data['price'], color="r")
plt.title("Distplot of Sale Price")
plt.ylabel("Number of Occurences")
plt.xlabel("Sale Price")
fig.add_subplot(3,2,2)
sns.boxplot(Data['price'])
plt.title('boxplot of Sale price')
plt.xlabel("Sale Price")
plt.ylabel("Number of Occurences")
fig.add_subplot(3,2,3)
sns.countplot(Data['price'], palette = 'Greens_d')
plt.title("Countplot of Sale Price")
plt.ylabel("Number of Occurences")
plt.xlabel("Sale Price")
fig.add_subplot(3,2,4)
plt.hist(Data['price'], bins=15,color="gold")
plt.title("Histplot of Sale Price")
plt.ylabel("Number of Occurences")
plt.xlabel("Sale Price")
plt.tight_layout()
fig.tight_layout(pad=1.5,w_pad=1.5,h_pad=10.0)
plt.show()
fig = plt.figure(figsize=(10,5))
fig.suptitle('scatterplot of salesprices', fontsize=14, fontweight='bold')
fig.add_subplot(3,3,1)
plt.scatter(Data["sqft_living"],Data.price, color='red')
plt.title("Sale Price wrt square Foot")
plt.ylabel('Sale Price')
plt.xlabel("sqft living ")
fig.add_subplot(3,3,2)
plt.scatter(Data["grade"],Data.price, color='orange')
plt.title("Sale Price wrt grade")
plt.ylabel('Sale Price ')
plt.xlabel("grade given to housing unit")
fig.add_subplot(3,3,3)
plt.scatter(Data["sqft_above"],Data.price, color='pink')
plt.title("Sale Price wrt squarefoot")
plt.ylabel('Sale Price')
plt.xlabel("sqft of house apart from basement")
fig.add_subplot(3,3,4)
plt.scatter(Data["bathrooms"],Data.price, color='purple')
plt.title("Sale Price wrt bathrooms")
plt.ylabel('Sale Price')
plt.xlabel("bathrooms ")
fig.add_subplot(3,3,5)
plt.scatter(Data["view"],Data.price, color='gold')
plt.title("Sale Price wrt view")
plt.ylabel('Sale Price')
plt.xlabel("view")
fig.add_subplot(3,3,6)
plt.scatter(Data["bedrooms"],Data.price, color='yellow')
plt.title("Sale Price wrt bedrooms")
plt.ylabel('Sale Price ')
plt.xlabel("bedrooms")
fig.add_subplot(3,3,7)
plt.scatter(Data["floors"],Data.price, color='green')
plt.title("Sale Price wrt floors")
plt.ylabel('Sale Price')
plt.xlabel("floors ")
fig.add_subplot(3,3,8)
plt.scatter(Data["sqft_basement"],Data.price, color='brown')
plt.title("Sale Price wrt square Foot")
plt.ylabel('Sale Price')
plt.xlabel("sqft of basement")
fig.add_subplot(3,3,9)
non_top_1_perc = Data.sort_values('price',ascending = False).iloc[216:]
sns.scatterplot(x='long',y='lat',data=non_top_1_perc,alpha = 0.8,palette = 'RdYlGn', hue='price')
plt.title("Sale Price per location ")
plt.ylabel('SalesPrice')
plt.xlabel('location')
plt.tight_layout()
fig.tight_layout(pad=1.5,w_pad=1.5,h_pad=10.0)
plt.show()
fig = plt.figure(figsize=(10,5))
fig.suptitle('countplot and distplot', fontsize=14, fontweight='bold')
fig.add_subplot(3,3,1)
sns.countplot(Data['bedrooms'], palette = 'Greens_d')
plt.title('no of bedrooms',loc='left')
plt.ylabel("Number of Occurences")
fig.add_subplot(3,3,2)
sns.countplot(Data['floors'],palette = 'Purples_r')
plt.title('no of floors',loc='left')
plt.ylabel("Number of Occurences")
fig.add_subplot(3,3,3)
sns.countplot(x='bathrooms',data=Data, palette = 'Oranges_r')
plt.ylabel("Number of Occurences")
plt.title('no of bathrooms',loc='left')
fig.add_subplot(3,3,4)
sns.distplot(Data["sqft_living"],color='r', kde=False);
plt.title("Distribution of Sqft_living")
plt.ylabel("Number of Occurences")
plt.xlabel("sqft_living");
fig.add_subplot(3,3,5)
sns.distplot(Data["sqft_basement"],color='purple', kde=False);
plt.title("Distribution of Sqft_basement")
plt.ylabel("Number of Occurences")
plt.xlabel('square feet of basement');
fig.add_subplot(3,3,6)
sns.distplot(Data['sqft_lot'],color='b',kde=False);
plt.title("Distribution of Sqft_lot")
plt.ylabel("Number of Occurences")
plt.xlabel('square feet of lot');
fig.add_subplot(3,3,7)
sns.distplot(Data["yr_built"],color='seagreen', kde=False);
plt.title("Distribution of year build")
plt.ylabel("Number of Occurences")
plt.xlabel('year bulit');
fig.add_subplot(3,3,8)
sns.distplot(Data["grade"].astype(int),color='r', kde=False);
plt.title("Distribution of  grade given")
plt.ylabel("Number of Occurences")
plt.xlabel('grade s=given to house');
fig.add_subplot(3,3,9)
sns.distplot(Data["waterfront"].astype(int),color='pink', kde=False);
plt.tight_layout()
fig.tight_layout(pad=1.5,w_pad=1.5,h_pad=10.0)
plt.show()
Data['date'] = pd.to_datetime(Data['date'])
Data['month'] = Data['date'].apply(lambda date:date.month)
Data['year'] = Data['date'].apply(lambda date:date.year)
fig = plt.figure(figsize=(10,5))
fig.add_subplot(1,2,1)
Data.groupby('month').mean()['price'].plot()
plt.title("house price vs months")
plt.ylabel("salesprice")
plt.xlabel('month');
fig.add_subplot(1,2,2)
Data.groupby('year').mean()['price'].plot()
plt.title("house price vs year")
plt.ylabel("saleprices")
plt.xlabel('year');
plt.tight_layout()
fig.tight_layout(pad=1.5,w_pad=1.5,h_pad=10.0)
plt.show()




