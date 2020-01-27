import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['stopwords','punkt','wordnet'])

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """Loads training data from Database

    Args:
    database_filepath: filepath for SQLite Database
    
    Returns:
    X: predictor values
    y: target values
    category_names: names of the target values
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('table_messages',engine)
    X = df.iloc[:,1]
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """tokenizes the text
    
    Args:
    text: text source
    
    Returns:
    tokens: list of tokenized text
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens



def build_model():
    """builds the model pipeline
         
    Returns:
    pipeline: pipeline of transformers and classifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # params dict to tune a model
    parameters = {
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_features': [None, 'log2', 'sqrt'],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [25, 100, 200],
    }

    # instantiate a gridsearchcv object with the params defined
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=4, n_jobs=6)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """calculates precision, recall and f1 scores
    
    Args:
    model: trained model
    X_test: test predictor values
    Y_test: test target values
    """
    Y_pred = model.predict(X_test) 

    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col],Y_pred[:,i]))


def save_model(model, model_filepath):
    """Saves model as pickl file

    Args:
    model: trained model
    model_filepath: file destination
    """
    output = open(model_filepath, 'wb')
    pickle.dump(model, output)
    output.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()