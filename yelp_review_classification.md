## Yelp Review Classification (WIP)

[Go back to the Main Page](index.md)

For this project, I will perform a **binary classification** on Yelp reviews and predict whether a review is positive or negative.

The dataset is obtained from the [Yelp Recruiting Competition on Kaggle](https://www.kaggle.com/c/yelp-recruiting) and contains 229,907 reviews. 


### Project Steps:
- Step 1: Extract Reviews and Ratings
- Step 2: Bag-of-Words Method
- Step 3: Word Embeddings and LSTM


### Step 1 & 2:

The data downloaded contains many features, including business type, date, name and so on. For this project, I only used the **reviews and the ratings** given by customers. With some operations using Pandas, the features are extracted.

The ratings are given on a scale of 1 - 5. Ratings 1 and 2 are negative labels, represented by 0, and ratings 4 and 5 are positive labels, represented by 1. Ratings of 3 are neutral and discarded. I computed the number of postive reviews and negative reviews and found out there are a lot more positive reviews than negative ones. I decided to only select some samples from the postive ones (the same number as negative reviews), because one it makes the dataset balanced and two it reduces the dataset size and would take less time to train. After sampling, I have 76946 reviews in total.

The first method I used is Bag-of-Words (BoW). BoW creates a frequency vector of tokens, which could be words or phrases, for a text document. The vector represents the number of occurances of each word in a text. Intuitively, we may expect to see complimentary words in positive reviews, such as "awesome", "useful" and "beatiful", while negative reviews should contain words like "worst", "broken", and "terrible". This means the counts for the corresponding words should differ for positive and negative reviews and thus are an important feature to use for classification.

Some words are fairly common or aren't useful for classification purposes, such as "the", "between" and "if". These words can be eliminated first. They are usually called stopwords in NLP. Same goes for most of the functuations.

These preprocessing techniques are implemented internally in scikit-learn for CountVecterizer(). 

TF-IDF, Term Frequency - Inverse Document Freqency, is a modified method. Instead of raw frequency of tokens, it uses "relative" frequency, in order to eliminate the common words across all text documents. TF-IDF gives a high value for tokens that have high frequency in certain documents but not all of them.

The raw frequency and TF-IDF both computes the frequency of tokens in texts. I used n-gram to specify how long a token can be (how many words). This is useful when we want to count phrases.

The computed frequencies are the features we can use in classification models. In this project, since it's a binary classification, I used logistic reggression classifier. 

The accuracy obtained on a test set was 0.92.

Check out [this Jupyter notebook](https://github.com/willchenyh/yelp_review_analysis/blob/master/Yelp%20Review%20Classification.ipynb) for details.


### Step 3:

Word embedding is a technique to represent words as vectors with their semantic features. The values in the vectors don't have concrete meanings (unlike the vectors in BoW represent frequency), but the words that have similar semantic meanings will have similar vectors - the distance between their vectors is short. For example, "apple" should be close to "banana" and far away from "dog".

The reason why we want to use word embedding in Recurrent Neural Networks (RNN) is that one it is embedded with semantic meanings, two we want to maintain the sequence of words from texts. RNNs are capable of learning sequential features from data.

RNN, as its name suggests, sends a neuron's output back to its input, as the neuron takes in a sequence. RNN can remember the past information in a sequence. LSTM, Long Short Term Memory, is a type of RNN that is good at remembering long term sequence features. An LSTM cell contains several layers, including "forget gate", "input gate", and "output gate". Intuitively, it can learn to forget the useless features and remember the useful ones.

In keras, embedding layer and LSTM layers are implemented and we just need to build a model with them. The last layer has one output to be either 0 or 1.

Check out [this program](https://github.com/willchenyh/yelp_review_analysis/blob/master/review_classification_lstm.py) for details


### References (Thank you!)
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/



### Picture Credit


