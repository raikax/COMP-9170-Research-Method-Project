# COMP-9170-Research-Method-Project
Fake News Detection and Media Trustworthiness: A Comparative Study of Explainable Machine Learning Models and Credibility Scoring

### Introduction
With the rise of social media and online news articles platforms, fake news has become a prevalent problem. Although manually checking every article is impractical, machine learning may help uncover patterns to separate true and false claims. In this project, we’ll compare Multinomial Naive Bayes and Logistic Regression for fake news detection, analyse which words influence their predictions, and assess if calibrated probabilities can act as a trustworthy indicator of an article’s veracity.

### Research questions
1. How well do Naive Bayes and Logistic Regression classify news articles as real or fake?

2. Which words have the greatest impact on the models’ predictions?

3. Can we use calibrated probabilities as a reliable indicator for fake news in individual articles?

4. How do biases in the data and different subjects influence the performance of the models?

### Data
For this project we use the Fake and Real News Dataset by Bisaillon (n.d.), available on Kaggle. It consists of 23,481 fake and 21,417 real news articles. Each article has a unique id and contains a title, subject, date and the text of the article. We found that there are no missing values but some duplicated articles in the dataset. After removing the completely duplicated articles we are left with 39,105 articles.

### Analysis
First, we’ll do some further exploratory data analysis, paying special attention to potential biases in the data. For example, we’ve already noticed that the subjects of the articles differ between real and fake news. If the models pick up on these differences it might skew their results.

After cleaning the data, we’ll turn the textual data into TF-IDF features and train a Multinomial Naive Bayes and Logistic Regression model on this data. We’ll evaluate the models’ performance using accuracy, precision, recall and F1-score. Next, we’ll investigate the words that have the largest (positive or negative) influence on the models’ predictions and do some probability calibration to see how reliable the models’ confidence is. If time permits, we’ll also compare the results to a DistilBERT model.

### Limitations
Although both models are likely to perform reasonably well, we expect their performance to differ somewhat. We also expect the models to rely heavily on certain words or subjects when making their predictions.

It’s likely that there are other biases present in the dataset, for example related to the source of the articles. It’s important to keep in mind that even if our model assigns a high probability to an article being real, this does not imply that all claims in the article are true.
