## House Price Predicition

[Go back to the Main Page](../index.md)

This is a regression problem with available features of houses to predict house prices. This page documents the steps and thinking process I took to solve this problem. I scored **top 10%** at the competition. 

The detailed code is in Jupyter Notebooks in [this repository](https://github.com/willchenyh/house_price_prediction/). I created a notebook for each project step as below.

Kaggle competition page: [https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### Project Steps

- Step 1: Understand Data
- Step 2: Create Baseline Score
- Step 3: Feature Engineering
- Step 4: Explore Linear Models
- Step 5: Explore Tree Based Models
- Step 6: Explore Stacked Models


### Step 1: Understand Data

The data information is given by the competition. The dataset has **2919** entries of vairous information of houses and their respective sale prices. The dataset is split into a training set of 1460 entries and a test set of 1459 entries. Each sample has **79** features, including diverse information, such as total square footage, number of bedrooms, and quality of materials. 


### Step 2: Create Baseline Score

I want to generate a baseline score using a basic model and mininum feature engineering. Future improvements can then be made based on this score.

The training and testing sets are **concatenated** so that data processing and transformation can be performed together. 

After a quick check, it was found that **34** out of 79 features contain **NA values**. A plot of percentage of NA values in features is shown below. Five features have more than 50% NA values. Sixteen features have only one or two NA values. The details and reasons for the NA values are to be explored. For now, I will simply remove all features with NA values.

![Percentage NA Plot](https://github.com/willchenyh/willchenyh.github.io/blob/master/house_price_prediction/pct_na_initial.png?raw=true)

After discarding features with NAs, there are **45** features remaining. Some of the features are numerical and others are categorical. Most of machine learning models, in this case regression models, can't deal with categorical data. Hence, we need to convert **categorical features** to numbers. **One-hot encoding** is the best method for the conversion. Here one thing to take note of is we shouldn't simply assign a different number to each category in a feature, because the numbers internally represent certain order which may not exist in the original feature. One-hot encoding creates a binary feature for each category and assigns "1" to this category and "0" to others. An example is shown below. With one-hot encoding, we now have **149** features.

![One hot encoding example](http://brettromero.com/wordpress/wp-content/uploads/2016/03/OHE-Image.png)

After basic data cleaning, we can now apply models. The error metric used is **RMSLE (root mean squared log error)** - the square root of mean squared error between log values of prediction and ground truth. The advantage of RMSLE over MSE (mean squared error) or RMSE is that with log this metric takes equal consideration for high and low prices. 

In order to obtain a general and accurate RMSLE from training data, **3-fold cross validation** is used. This tools splits the training set into three small sets, and uses one of them as the test set and the rest as training set. The mean RMSLE of all five small test sets is computed.

I picked the most basic **linear regression** model (available in sklearn) first. However, this model raised an error indicating that the prediction contains "infinity" values, which is probably due to the unstableness of the model. Hence, I tried **linear regression with l2-norm regularization** (available in sklearn as **Ridge**) to limit the coefficients. Ridge ran with no issues and gave an RMSLE of **0.1759**, which is better than 1/3 of the competition participants. 


### Step 3: Feature Engineering

Now let's take a closer look of the features. The main goal of this step is to remove outliers, understand and fill in the NA values, extract and create new features, and transform features if necessary. 

#### Outliers

Let's start with **outliers**. The dataset [documentation](https://ww2.amstat.org/publications/jse/v19n3/decock.pdf) suggests removing the data points with **GrLivArea larger than 4000**. As we can see from a plot of SalePrice vs GrLivArea plot below, there are four points greater than 4000. Two points at bottom right are true outliers as they are very far away from the main cluster. They represent two houses with large square footage but very cheap prices - possibly they were sold under unusual circumstances. The two points at top right, although far from the main cluster, follow the relation of the two variables based on the main cluster. Nevertheless, they are removed based on the dataset documentation.

![SalePrice vs GrLivArea](https://github.com/willchenyh/willchenyh.github.io/blob/master/house_price_prediction/saleprice_vs_grlivarea.png?raw=true)

#### Filling in NAs

As for the **NA values**, after cross-comparing the data and the provided description, I noticed that most of the NA values are probably not "missing data" but contain meaningful information - **the absence of certain feature in the respective houses**. For example, BsmtQual uses NA to indicate "No basement", and the same goes for GarageType, Alley and so on. For these features, NA values are replaced with "None".

Other features are not as straight-forward. How they are filled in are shown in table below.

**Feature** | **Method to fill in NA**
--- | ---
LotFrontage | Use medium of the same neighborhood
Utilities | The test set only has Allpub value, and it has only 2 missing values. It's insignificant, so the feature is removed
Basement related features (numerical) | Use 0
Functional | Use "Typ"
Garage related features (numerical) | Use 0
Other features | Use mode  
 | 
   
   
   
Now all of the NA values are taken care of, we can start more advanced feature engineering. **Some numerical features are supposed to be categorical**, such as MSSubClass, which is the house type. **Some categorical features make more sense with ordinal values**, which I will convert to numbers based on the ratings. For example, PoolQC (pool quality) has values None, Fair, Average, Good, Excellent, and they will be converted to 0, 1, 2, 3, 4 respectively. The reason for this conversion is that the ordinal numbers are meaningful for the regression models we use. 

We can also **create new features** based on the existing features. First, some house functionalities have both quality and condition ratings. They can be multiplied to produce overall scores. Second, total numbers can be produced from multiple features. For example, total square footage of a house can be computed with its basement square footage and above-ground living area. These operations increases the total number of features from 79 to **91**.

Okay. Now the Ridge model gives a better score of **0.1563** - feature engineering is helpful!


### Step 4: Explore Linear Models

In order to improve the prediction score, we can try different models. Linear regression models are a good place to start. 

**Linear models** are intuitive - they form linear relations between independent features and the target variable. For this problem, intuitively, we would hypothesize that house prices should follow a linear relation with features. For example, a house with two bedroom and a pool should be more expensive than a house with one bedroom and no pool.

The formula of the most basic linear regression and an example plot are shown below. The models will compute the best values of w's (the **coefficients of features**). The models are usually evaluated with MSE, as linear regression essentially can be solved as a least squares problem.

![linear regression formula](http://scikit-learn.org/stable/_images/math/334dd847bce79ed52a760f02b3efd8faefdb6e8b.png)

![linear regression example](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/400px-Linear_regression.svg.png)

#### Log Transformation

Linear regression has some **assumptions** for it to work properly. They are listed on [the Wikipedia page](https://en.wikipedia.org/wiki/Linear_regression#Assumptions). The assumption of **homoscedasticity** is interesting to check. Homoscedasticity is a property of constant variance, meaning the residual (difference of prediction and truth value) variance should be constant for small and large values of the target variable. This can be obtained from **residuals of a normal distribution**, which happens when variables are normal distributions. Sometimes variables are skewed normal distributions and with the help of some transformations they can be converted to Gaussian. 

In order to check for skewness, we can use the skew function in scipy. A positive return value indicate a right-skewed distribution (right tail is longer) and a negative return value indicate a left-skewed distribution. A right-skewed distribution can be converted to Gaussian with a **log transformation**. It turns out that the target variable SalePrice is right skewed. I log-trainsformed SalePrice and the result is shown below:

![saleprice plot](https://github.com/willchenyh/willchenyh.github.io/blob/master/house_price_prediction/saleprice.png?raw=true)
![saleprice log transformed plot](https://github.com/willchenyh/willchenyh.github.io/blob/master/house_price_prediction/saleprice_log.png?raw=true)

Looks pretty good! Now we can do the same for the independent variables. I set the skewness threshold as 0.6 - I will apply log transformation if the skewness is greater than 0.6. It was found that 35 out of 66 numerical features are right skewed.

Now let's move on to exploring different models.

#### Regularization

We encountered a problem in step 2 when we used a simple linear regression model. The issue was probably caused by large coefficients, which can be a common issue for linear regression models as the coefficient sizes are not bounded. Therefore, it is often encouraged to use regularization with models to limit the coefficient sizes. Two common **regularization techniques** are L1-norm and L2-norm. 

The **L1-norm regularization** for linear regression is available in sklearn as **Lasso**, whose formula is shown below:

![lasso formula](http://scikit-learn.org/stable/_images/math/07c30d8004d4406105b2547be4f3050048531656.png)

Lasso adds the **absolute value of coefficients** as penalty to the formula to be minimized, which prevents the coefficients to become too large. Running Lasso with default function parameters gives us a prediction score of **0.2518**. Hmmm this is much worse than our previous score using Ridge - should we eliminate this model? Wait! This score was obtained using default function parameters. We should always tune the function parameters for specific problems. Maybe we will get a good score afterwards.

The Lasso function has a parameter **alpha**, which is the coefficient of the penalty w in the loss function mentioned above. We can use **grid search** (available in sklearn) to find the optimal alpha value. Turns out the best alpha is **0.0005** and the new prediction score is **0.1106**. Wow, thats a huge improvement!

Another advantage Lasso has is that it works well in sparse feature space by eliminating useless features. Out of 258 features, only **98 have non-zero coefficients**, meaning only 98 features are considered useful.

The **L2-norm regularization** for linear regression is available in sklearn as **Ridge**, whose formula is shown below:

![ridge formula](http://scikit-learn.org/stable/_images/math/48dbdad39c89539c714a825c0c0d5524eb526851.png)

Ridge adds the **squared value of coefficients** as penalty to the formula to be minimized, which also prevents the coefficients to become to large. They should have similar effects of regularization intuitively and as for which one is better really depends on specific problems. Using grid search, the best alpha obtained is 8 and prediction score is **0.1134**.

There is another model with L2 regularication called **Kernel Ridge Regression**. I will use polynomial kernel, which makes the model behave as a **polynomial regression model**. Polynomial regression takes in features in their nth degree polynomials and creates a more complex relation between features and target variable. Turns out it gives a higher error than Ridge, with **0.1158**.

We can also use both L1 and L2 regularization, which is available in sklearn as ElasticNet. The formula is shown below:

![elasticnet formula](http://scikit-learn.org/stable/_images/math/51443eb62398fc5253e0a0d06a5695686e972d08.png)

Using grid search, the best parameters are alpha=0.0005 and l1_ratio=0.9, and the prediction score is **0.1106**.

#### Preprocessing Technique

Sometimes it is helpful to preprocess data before applying models. Usually the features are scaled to zero mean and unit variance. Another way is to scale based on median and interquartile range (available in sklearn as RobustScaler). The second method will have less effect from outliers (extreme values). After I applied this technique, all of the models showed improvements. Lasso and ElasticNet have minimal improvement (on a scale of 10^-5); Ridge and Kernel Ridge made improvements on a scale of 10^-3.

#### Overall

Overall, the best linear model is Lasso with a score of 0.1105.


### Step 5: Explore Tree Based Models

Another common group of models is the **tree based models**. In this step, I will go over Decision Tree, Random Forest, and Gradient Tree Boosting. 

If you are familiar with data structures, you've probably heard of Binary Search Tree. For BST in the case of numbers, using the root number as a threshold, a smaller number goes to the left branch, and a larger number goes to the right. Similarly for a **Decision Tree**, it splits data based on thresholds of feature values. Each leaf contains a small collection of data points that satisfies all the rules going down its branch. When a new data point to be predicted is introduced to a trained decision tree, it will go through the rules and land on the leaf that best fit the point's feature values. The predicted value will then be the mean of the leaf points. 

Here's a graph example of a Decision Tree.

![decision tree](https://i.ytimg.com/vi/ydvnVw80I_8/maxresdefault.jpg)

The Decision Tree, after grid search, gives a prediction score of **0.1987**, which is even worse than our baseline. 

The results of a Decision Tree model really depends on the rules generalized for splitting branches. Chances are you may get different splitting rules if you run a deep model multiple times. In order to make a more general model, **Random Forest** is a good choice. The name tells you what a Random Forest is - a group of Decision Trees. The advantage of Random Forest is that by averaging over the results of all the Decision Trees, it can **reduce the variance** of a Decision Tree model and give a generally more accurate result. Theoretically, one would be incentivized to use deep trees (low bias and high variance) over shallow trees (high bias and low variance).

![random forest](https://s3.ap-south-1.amazonaws.com/techleer/113.png)

The Random Forest gives a much better result of **0.1352** with 3000 Trees. It generalizes much better than a single Decision Tree. I used up to 3000 trees in grid search because it takes a while to run large number of trees. It can possibly perform better if using more Trees, but I don't think it would better than the linear models we used since from 1000 to 3000 the improvement was small, on a scale of 10e^-3.

On the other hand, **Gradient Tree Boosting** "stacks" multiple Trees on top of each other - single Decision Trees run on the values produced by previous Trees. This model utilizes **Gradient Descent** method to follow the gradient of the given loss function. Gradient Descent is an iterative method to find minimum point of a loss function, which is MSE for this problem. Intuitively, for each Tree in the model, we step forward in the direction of gradient computed by the previous Tree. The gradient is computed as the mean of gradients of data points in a leaf. Hence, The Gradient Tree Boosting is a complex model to **reduce bias**. Theoretically, one would be incentivized to use shallow trees (high bias and low variance).

The Gradient Tree Boosting gives an even better score of **0.1188**. As hypothesized, it worked best with shallow trees of two levels. 


### Step 6: Explore Stacked Models

After some experiments with individual models, it might be interesting to see the results of combined models. The values predicted by various models are slightly different. One model might give very accurate predictions for certain points and another for other points. Hence, if we combine their predictions, we may get more accurate results overall. 

First, let's try a simple **"averaging model"**, which uses the mean of predictions of all models. Since this model is not readily available, I will go ahead and write it on my own. Trust me, it's pretty straight-forward. I'm using Lasso, Ridge, ElasticNet and Gradient Tree Boosting as the base models. The averaging model gives the best score so far of **0.1084**!

Next, I want to try a more complex model - let's just call it **stacked model**. Instead of just taking the mean of predictions, this model will try to train a meta model to find the best relationship between the predictions and the ground truth. It's a little trickier to implement than the averaging model, because it needs to learn two levels of models during training. 

We need to split the training set into three subsets (four or five are also fine - just follow your cross validation convention), say A, B and C. Take A as the first test subset and combine B and C as the training subset. Use [B,C] to train base models (Lasso, Ridge, ElasticNet and Gradient Tree Boosting) and predict on A. Then use [A,C] to predict B, and use [A,B] to predict C. Now we have the predictions for the entire training set, so we can move on to train the meta model. I'm using Lasso model for this case. This stacked model gives a score of **0.1094** - also pretty good!

This model works well on data that contains complex features. Make sure you don't overfit.


### Conclusion

This has been an interesting project to explore. I'm sure there are more techniques we can use - and I'm still learning as well!

Let's me know if you have any thoughts/suggestions/comments! I'd be happy to discuss and experiment!


### References (Thank you!)
https://www.kaggle.com/apapiu/regularized-linear-models
https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/


### Picture Credit
https://s3.ap-south-1.amazonaws.com/techleer/113.png
https://i.ytimg.com/vi/ydvnVw80I_8/maxresdefault.jpg
http://scikit-learn.org/stable/_images/math/51443eb62398fc5253e0a0d06a5695686e972d08.png
http://scikit-learn.org/stable/_images/math/48dbdad39c89539c714a825c0c0d5524eb526851.png
http://scikit-learn.org/stable/_images/math/07c30d8004d4406105b2547be4f3050048531656.png
https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/400px-Linear_regression.svg.png
http://scikit-learn.org/stable/_images/math/334dd847bce79ed52a760f02b3efd8faefdb6e8b.png
http://brettromero.com/wordpress/wp-content/uploads/2016/03/OHE-Image.png

