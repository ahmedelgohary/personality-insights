# surrounding-me

---
[![Build Status](https://travis-ci.com/ahmedelgohary/surrounding-me.svg?token=bg9PLDmYCBsbE4hyvqF8&branch=master)](https://travis-ci.com/ahmedelgohary/surrounding-me)


Surrounding me is a program that utilizes machine learning to determine your friends' personalities, using the Myers–Briggs Type Indicator (MBTI). These personalities consist of 4 letters, where each letter could be one of two options, Introversion (I) vs Extraversion (E), Sensing (S) vs Intuition (N), Thinking (T) vs Feeling (F) and Judging (J) vs Perceiving (P). This gives us 2^4 personality types, or 16 different personalities. 


Some metrics about the MBTI dataset used:
https://www.kaggle.com/datasnaek/mbti-type

We have: 
8675 rows corresponding to 8675 people with different personality types. 
2 columns, one for the person's personality type and one for a set of texts that person sent.

Some metrics:
1) 16 topics, the MBTI personality type has 16 different personality types.
2) Very unbalanced dataset: 
    INFP    1832
    INFJ    1470
    INTP    1304
    INTJ    1091
    ENTP     685
    ENFP     675
    ISTP     337
    ISFP     271
    ENTJ     231
    ISTJ     205
    ENFJ     190
    ISFJ     166
    ESTP      89
    ESFP      48
    ESFJ      42
    ESTJ      39

3) A median of 1278 words per post for each person
4) We split the data into 80:20, with 80% of the data being used for training and 20% for testing



Since our number of samples is 8675, with around 1278 words per sample, our number of samples/number of words per sample ratio is around 7. Hence, we will be tokenizing the text as n-grams and using a multi-layer perceptron to classify them. An n-gram model will handle the texts as different "sets" of words. 

This is done by:
1) Tokenizing the data into word unigrams and bigrams
2) Vectorizing using term frequency - inverse document frequency (tf-idf) encoding
3) Selecting the top 20,000 features from the vector by getting rid of rare tokens, and using f_classif to get the most important features

Since we have multi-class classification, we will be using softmax as our activation function and categorical cross entropy as our loss function. Dropout regularlization will be used to train the neural network, in order to prevent overfitting in our classification.