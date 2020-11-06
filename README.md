# house_price_prediction
PREDICTING PRICE OF HOUSES BY ANALYSING VARIOUS FACTORS

  This project applies linear regression on dataset to predict the selling price of a new home.The objective of the project is to perform data visulalization techniques to understand the insight of the data. This project aims apply various Python tools to get a visual understanding of the data and clean it to make it ready to apply machine learning opertation on it.
Software and Libraries
This project uses the following software and Python libraries:
* Python
* NumPy
* pandas
* matplotlib
* seaborn 
* sklearn
CODE EXPLANATION:
  First import our libraries and dataset .The Kaggle House Prices datasets can be downloaded from kaggle.com/harlfoxem/housesalesprediction

     
    
    
By using Pandas.read_csv(), read our dataset and viewing our dataset using Pandas.head().T that will transpose our columns and rows
    
    Data = pd.read_csv('kc_house_data.csv')
    print(Data.head(5).T)
    Data.info()
    print(Data.describe())

Pandas.info() function is used to get a concise summary of the dataframethat  includes list of all columns with their data types and the number of non-null values in each column. we also have the value of rangeindex provided for the index axis.Pandas describe() is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values.

 #DATA CLEANING   
 
    Data = Data.drop('id',axis=1)
    Data = Data.drop('zipcode',axis=1)
    print(Data.isnull().sum())
    
 Drop 'id' and 'zipcode' columns from a Data using Pandas.drop() and Pandas isnull() function detect missing values in the given object. It return a boolean same-sized object indicating if the values are NA. Missing values gets mapped to True and non-missing value gets mapped to False and Pandas.isnull().sum() adds no of true values

#DATA VISUALISATION    
    
    fig = plt.figure(figsize=(7,6))
    fig.add_subplot(1,2,1)
    sns.distplot(Data['price'],color="y")
    plt.title('distplot of price')
    fig.add_subplot(1,2,2)
    sns.boxplot(Data['price'])
    plt.title('boxplot of price')
    plt.show()

The purpose of using plt.figure() is to create a figure object and using figsize , able to resize that figure 
sns.distplot() combines the matplotlib hist function with the seaborn kdeplot() and rugplot() functions. kdeplot() represents the data using a continuous probability density curve in one or more dimensions and rugplot()  is intended to complement other plots by showing the location of individual observations in an unobstrusive way.
sns.boxplot() returns the five number summary, which is the minimum, first quartile, median, third quartile, and maximum.

#VISUALIZATION OF NO OF BEDROOMS,BATHROOMS,FLOORS AND DATES TO BE SOLD GRADES GIVEN AGAINST PRICES
    
    fig = plt.figure(figsize=(12,12))
    fig.add_subplot(3,2,1)
    sns.countplot(Data['bedrooms'], palette = 'Greens_d')
    plt.title('no of bedrooms',loc='left')
    fig.add_subplot(3,2,2)
    sns.countplot(Data['floors'])
    plt.title('no of floors',loc='left')
    fig.add_subplot(3,2,3)
    sns.countplot(x='bathrooms',data=Data)
    plt.title('no of bathrooms',loc='left')
    fig.add_subplot(3,2,4)
    sns.countplot(Data['grade'])
    plt.title('grade given to ousing unit', loc='left')
    Data['date'] = pd.to_datetime(Data['date'])
    Data['month'] = Data['date'].apply(lambda date:date.month)
    Data['year'] = Data['date'].apply(lambda date:date.year)
    fig.add_subplot(3,2,5)
    Data.groupby('month').mean()['price'].plot()
    plt.title('month', loc='left')
    fig.add_subplot(3,2,6)
    Data.groupby('year').mean()['price'].plot()
    plt.title('year', loc='left')
    fig.tight_layout(pad=2.0 ,w_pad=1.5, h_pad=10.0)
    plt.tight_layout()
    plt.show()
sns.countplot() shows the number of occurrences of an item based on a certain type of category.Pandas.apply(lambda date:date.month)extracts month and as Pandas.apply(lambda date:date.year).By using Data.groupby('month').mean()['price'].plot(), plot the mean of months against prices and as  Data.groupby('year').mean()['price'].plot()

#VISUALIZATION OF SQUARE FOOTAGE OF HOUSE ,LIVING ROOM,BASEMENT,LOT,LOCATION OF HOUSES AGAINST PRICES
    
    fig = plt.figure(figsize=(12,12))
    fig.add_subplot(3,2,1)
    sns.scatterplot(Data['sqft_above'], Data['price'])
    plt.title('sqftft of house ',loc='right',fontsize=10)
    fig.add_subplot(3,2,2)
    sns.scatterplot(Data['sqft_lot'],Data['price'])
    plt.title('sqftft of lot',loc='right',fontsize=10)
    fig.add_subplot(3,2,3)
    sns.scatterplot(Data['sqft_living'],Data['price'])
    plt.title('sqftft of home',loc='right',fontsize=10)
    fig.add_subplot(3,2,4)
    sns.scatterplot(Data['sqft_basement'],Data['price'])
    plt.title('sqft ft of basement',loc='right',fontsize=10)
    non_top_1_perc = Data.sort_values('price',ascending = False).iloc[216:]
    fig.add_subplot(3,2,5)
    sns.scatterplot(x='long',y='lat',data=non_top_1_perc,alpha = 0.8,palette = 'RdYlGn', hue='price')
    plt.title('loc of house vs prices',loc='left')
    #sns.pairplot(Data['bedrooms bathrooms sqft_living floors waterfront price'.split()])
    fig.tight_layout(pad=2.0 ,w_pad=1.5, h_pad=10.0)
    plt.tight_layout()
    plt.show()
    
 A scatter plot is a diagram where each value in the data set is represented by a dot.Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe.Pandas.sort_values() sort the prices values in descending order.Data.sort_values('price',ascending = False).iloc[216:] returns dataset upon descending order of prices except first 216 values.The pairplot function creates a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column.

#SPLITING THE DATASET

     Data = Data.drop('date',axis=1)
     X = Data.drop('price',axis =1).values
     y = Data['price'].values
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
 Drop date column in dataset and includes all columns except price in X and icludes values of prices in Y . split the datasets into traning set and test set
 
#STANDARDIZATION OF DATASETS 

     s_scaler = StandardScaler()
     X_train = s_scaler.fit_transform(X_train.astype(np.float))
     X_test = s_scaler.transform(X_test.astype(np.float))
      
StandardScaler performs the task of Standardization and s_scaler.fit_transform ()fit  data, then transform float type.

# METHOD 1:
# MULTIPLE LINEAR REGRESSION
   Linear Regression refers to a model that assumes a linear relationship between input variables and target variables.Linear regression attempts to model the relationship between two variables by fitting a linear equation to observed data.
   Multiple Linear Regression is an extension of Simple Linear Regression.Multiple regression generally explains the relationship between multiple independent or predictor variables and one dependent or criterion variable.
# METHOD 2:
# RIDGE REGRESSION 
  With increase of no of features,regression equation becomes higher order(polynomial equation)and leads to overfit that mean doesn't fit for realtime values and eliminate overfit by reducing size of coefficients 
  Ridge Regression is a popular type of regularized linear regression that includes an L2 penalty and  an extension of linear regression that adds a regularization penalty to the loss function during training.L2 penalty is to penalize a model based on the sum of the squared coefficient values (alpha). 
# METHOD 3:
# LASSO REGRESSION 
 Lasso Regression uses L1 regularization technique and also  extension of linear regression that adds a regularization penalty to the loss function during training.L1 penalty is to penalize a model based on the absolute of the squared coefficient values (alpha).
 # METHOD 4:
# ELASTIC NET REGRESSION 
  By combining lasso and ridge regression we get Elastic-Net Regression. Elastic Net is proved to better it combines the regularization of both lasso and Ridge
# METHOD 2
# GRADIENTBOOSTING REGRESSION
   Gradient boosting is one of the most powerful techniques for building predictive models and It is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
 
   
we have predicted the house price using different ML model algorithms.
 
