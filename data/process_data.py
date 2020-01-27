import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads message and category data

    Args:
    messages_filepath: filepath for message csv file
    categories_filepath: filepath for categories csv file

    Returns:
    df: DataFrame with merged information from source files
    """

    messages = pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """Cleans category data and removes duplicates

    Args:
    df: DataFrame from load_data function

    Returns:
    df: DataFrame with cleaned data
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = list(categories.iloc[0])
    # use this row to extract a list of new column names for categories.
    category_colnames = [i.split('-')[0] for i in row]
    # allocate new column names
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(str)

    #drop original categories column
    df.drop('categories',inplace=True, axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1).reindex(df.index)
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)

    return df


def save_data(df, database_filename):
    """Saves DataFrame to SQLite Database

    Args:
    df: cleaned DataFrame from clean_data function
    database_filename: path for database
    """

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('table_messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
