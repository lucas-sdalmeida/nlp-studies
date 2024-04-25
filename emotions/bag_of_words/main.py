import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

reviews = pd.read_csv('../assets/imdb-reviews-pt-br.csv')
# Creating a new column in the review with binary representations of the results
reviews['classification'] = reviews['sentiment'].transform(lambda x: 0 if x == 'neg' else 1)

# CountVectorizer is a sklearn tool to get a bag of words from a text array
vectoring = CountVectorizer(max_features=50)
bag_of_words = vectoring.fit_transform(reviews['text_pt'])
# print(f'{bag_of_words.shape = }')

# Split the train and test classes and training a LogistRegression model
x_train, x_test, y_train, y_test = train_test_split(bag_of_words, reviews['classification'], random_state=42)
regression = LogisticRegression()
regression.fit(x_train, y_train)
accuracy = regression.score(x_test, y_test)
print(f'{accuracy = }')
