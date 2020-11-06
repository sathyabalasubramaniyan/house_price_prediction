# house_price_prediction
PREDICTING PRICE OF HOUSES BY ANALYSING VARIOUS FACTORS
     
 The project applies regression algorithmns on dataset to predict the selling price of a new home.The objective of the project is to perform data visulalization techniques to understand the insight of the data.This project evaluate the performance and the predictive power of a model trained and tested on data set 
 
 The dataset contains 21613 entries and 21 variables.
         
       price            
       sqft_living     
       grade            
       sqft_above       
       sqft_living15    
       bathrooms        
       view             
       sqft_basement    
       bedrooms        
       lat              
       waterfront       
       floors           
       yr_renovated  
       sqft_lot      
       sqft_lot15    
       yr_built        
       condition       
       long           
       year          
       month           
   I have implemented  this project in two parts. First part contains data analysis by visualization of datasets in various plot as explained in Datavisual.py. Second is training of machine learning models explained in Houseprediction.py  
   Here, I have trained variousalgorithms like
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
 Lasso regression is a type of linear regression that uses shrinkage. Shrinkage is where data values are shrunk towards a central point, like the mean.Lasso Regression uses L1 regularization technique and also  extension of linear regression that adds a regularization penalty to the loss function during training.L1 penalty is to penalize a model based on the absolute of the squared coefficient values (alpha).The goal of lasso regression is to obtain the subset of predictors that minimizes prediction error for a quantitative response variable.
 # METHOD 4:
# ELASTIC NET REGRESSION 
  By combining lasso and ridge regression we get Elastic-Net Regression. Elastic Net is proved to better it combines the regularization of both lasso and Ridge.The benefit is that elastic net allows a balance of both penalties, which can result in better performance than a model with either one or the other penalty on some problems.
# METHOD 5
# GRADIENTBOOSTING REGRESSION
   Gradient boosting is one of the most powerful techniques for building predictive models and produces a prediction model in the form of an ensemble of weak prediction models. Gradient boosting Regression calculates the difference between the current prediction and the known correct target value.
This difference is called residual. After that Gradient boosting Regression trains a weak model that maps features to that residual. This residual predicted by a weak model is added to the existing model input and thus this process nudges the model towards the correct target. Repeating this step again and again improves the overall model prediction..
 
# CONCLUSION     
  we have predicted the house price using different ML model algorithms.
Gradient boosting regression  is the model that performed best among the several models ( MULTIPLE LINEAR REGRESSION,RIDGE REGRESSION,LASSO REGRESSION,ELASTIC NET
REGRESSION ) 
The performance of GBR is Train score of 97% and test score of 89%
 
