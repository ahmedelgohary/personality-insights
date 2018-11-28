# Personality Insights
[![Build Status](https://travis-ci.com/ahmedelgohary/personality-insights.svg?branch=master)](https://travis-ci.com/ahmedelgohary/personality-insights)

---
## Background info:
Personality insights is a program that utilizes machine learning to determine your friends' personalities, using the [Myers–Briggs Type Indicator (MBTI)](https://www.myersbriggs.org/my-mbti-personality-type). 

These personalities consist of 4 letters, where each letter could be one of two options:

 Introversion (I) vs Extraversion (E)
 
 Sensing (S) vs Intuition (N)
 
 Thinking (T) vs Feeling (F) 
 
Judging (J) vs Perceiving (P). 

This gives us 2^4 personality types, or 16 different personalities. 

---
## Dataset and Metrics
[This dataset](https://www.kaggle.com/datasnaek/mbti-type) from Kaggle was used. 


Our dataset contains 8675 rows corresponding to 8675 people with one column for the person's personality type and one for a set of texts that person sent.

Some metrics:
1) We have 16 topics since the MBTI personality type classifies everyone into one of 16 different personality types.
2) A very unbalanced dataset: 
   
    Personality | Count
    --- | ---
    INFP | 1832
    INFJ  |  1470
    INTP   | 1304
    INTJ   | 1091
    ENTP   |  685
    ENFP   |  675
    ISTP   |  337
    ISFP   |  271
    ENTJ   |  231
    ISTJ   |  205
    ENFJ   |  190
    ISFJ   |  166
    ESTP   |   89
    ESFP   |   48
    ESFJ   |   42
    ESTJ   |   39

3) A median of 1278 words per post for each person
4) We split the data into 80:20, with 80% of the data being used for training and 20% for testing



Since our number of samples is 8675, with around 1278 words per sample, our number of samples/number of words per sample ratio is around 7. 

Since this ratio is pretty small, tokenizing the text as n-grams (unigrams and bigrams here) and using a multi-layer perceptron to classify them will be a better approach than tokenizing the texts as a sequence. An n-gram model will handle the texts as different a "bag of words".

To summarize, data preprocessing is done by the following:
1) Tokenizing the data into word unigrams and bigrams
2) Vectorizing using term frequency - inverse document frequency (tf-idf) encoding
3) Selecting the top 20,000 features from the vector by getting rid of rare tokens, and using f_classif to get the most important features


Since we have multi-class classification, we will be using **softmax** as our activation function and **sparse categorical cross entropy** as our loss function. **Dropout regularlization** will be used to train the neural network, in order to prevent overfitting in our classification, with a 20% dropout rate at each layer.
