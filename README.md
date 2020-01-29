README

# disaster_response_pipeline

### Table of Contents

1. [Installation](#installation)
2. [Project Description](#description)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Additionally to the Anaconda distribution of Python it is necessary to install plotly library. The code should run with no issues using Python versions 3.*.

    1. Run the following command in the app's directory to run your web app.
        `python run.py`

    2. Go to http://0.0.0.0:3001/

## Project Description <a name="description"></a>

### Business Understanding

In case of an emergency every minute counts. A Machine Learning model can classify messages in Social Media and help emergency worker to make better and faster decisions. This enables better coordination for versatile needs as food, accommodation or medical help.

### Data Understanding

Emergency messages for this analysis project were provided by Figure Eight. They come with 36 categories, which serve as our target for the machine learning model.

### Data Preparation and Modeling

To analyze tweets, the text is cleaned from punctuation and stopwords, stemmed and tokenized.

### Result Evaluation

The web app shows promising results classifying new messages.

## File Descriptions <a name="files"></a>

The results can be viewed in a Flask Web App, which imports the created model to classify new messages and displays visualizations of the training dataset.  


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data was provided by Figure Eight in context of a Udacity online course.
