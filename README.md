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
  
 
    Data = pd.read_csv('kc_house_data.csv')
     
    
    
By using Pandas.read_csv(), read our dataset and viewing our dataset using Pandas.head().Tthat will transpose our columns and rows

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
    
    fig = plt.figure(figsize=(10,7))
    fig.add_subplot(2,1,1)
    sns.distplot(Data['price'])
    fig.add_subplot(2,1,2)
    sns.boxplot(Data['price'])
    plt.show()

The purpose of using plt.figure() is to create a figure object and using figsize , able to resize that figure 
sns.distplot() combines the matplotlib hist function with the seaborn kdeplot() and rugplot() functions. kdeplot() represents the data using a continuous probability density curve in one or more dimensions and rugplot()  is intended to complement other plots by showing the location of individual observations in an unobstrusive way.
sns.boxplot() returns the five number summary, which is the minimum, first quartile, median, third quartile, and maximum.

METHOD 1:
LINEAR REGRESSION
 
