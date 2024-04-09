import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Opening the reviews csv and creating a data frame of it
reviews = pd.read_csv('assets/imdb-reviews-pt-br.csv')

# Splitting the reviews data frame into proper arrays to train the machine learning model
x_train, x_test, y_train, y_test = train_test_split(reviews, reviews.sentiment, random_state=42)

# Creating and training the model and getting its score
regression = LogisticRegression()
regression.fit(x_train, y_train)
accuracy = regression.score(x_test, y_test)

print(f'{accuracy = }')

# The code above is meant to not work. The expected error means that machine learning models cannot understand natural
# language directly, it is needed a different approach to train these models: Natural Language Processing.
