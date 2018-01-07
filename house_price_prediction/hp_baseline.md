## House Price Predicition

This is a regression problem with available features of houses to predict house prices. This page documents the steps and thinking process I took to solve this problem. I scored **top 10%** at the competition. 

The detailed code is in Jupyter Notebooks in this repository.

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

In order to obtain a general and accurate RMSLE from training data, **5-fold cross validation** is used. This tools splits the training set into five small sets, and uses one of them as the test set and the rest as training set. The mean RMSLE of all five small test sets is computed.

I picked the most basic **linear regression** model (available in sklearn) first. However, this model raised an error indicating that the prediction contains "infinity" values, which is probably due to the unstableness of the model. Hence, I tried **linear regression with l2-norm regularization** (available in sklearn as **Ridge**) to limit the coefficients. Ridge ran with no issues and gave an RMSLE of **0.1759**, which is better than 1/3 of the competition participants. 

### Step 3: Feature Engineering

Now let's take a closer look of the features. The main goal of this step is to remove outliers, understand and fill in the NA values, extract and create new features, and transform features if necessary. 

Let's start with **outliers**. The dataset [documentation](https://ww2.amstat.org/publications/jse/v19n3/decock.pdf) suggests removing the data points with **GrLivArea larger than 4000**. As we can see from a plot of SalePrice vs GrLivArea plot below, there are four points greater than 4000. Two points at bottom right are true outliers as they are very far away from the main cluster. They represent two houses with large square footage but very cheap prices - possibly they were sold under unusual circumstances. The two points at top right, although far from the main cluster, follow the relation of the two variables based on the main cluster. Nevertheless, they are removed based on the dataset documentation.

![SalePrice vs GrLivArea](https://github.com/willchenyh/willchenyh.github.io/blob/master/house_price_prediction/saleprice_vs_grlivarea.png?raw=true)

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

Now all of the NA values are taken care of, we can start more advanced feature engineering. **Some numerical features are supposed to be categorical**, such as MSSubClass, which is the house type. **Some categorical features make more sense with ordinal values**, which I will convert to numbers based on the ratings. For example, PoolQC (pool quality) has values None, Fair, Average, Good, Excellent, and they will be converted to 0, 1, 2, 3, 4 respectively. The reason for this conversion is that the ordinal numbers are meaningful for the regression models we use. 

We can also **create new features** based on the existing features. First, some house functionalities have both quality and condition ratings. They can be multiplied to produce overall scores. Second, total numbers can be produced from multiple features. For example, total square footage of a house can be computed with its basement square footage and above-ground living area. These operations increases the total number of features from 79 to **91**.

Okay. Now the Ridge model gives a better score of 0.1563 - feature engineering is helpful!

### Step 4: Explore Linear Models




### Markdown (Example)

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
```



