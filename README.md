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

Airbnb, founded as part of the sharing economy, enables everyone to rent out accommodations for short term. After it’s expansion in the last decade, the company faces criticism as rising Airbnb rentals extract living spaces and leads to higher rents for locals.

The analysis should show if Airbnb is a serious competitor to the hospitality industry, who offers accommodations and how cities are impacted from this development.

### Data Understanding

This analysis project uses data from insidedairbnb. Key objects are Airbnb listings and their reviews, which are provided in separated files. Airbnb states that about 50% of guests leave a review, which makes them a good proxy for the platform usage in a city. The listing file consists of one row per listing and contains an ID, listings and hosts name as several aggregated values. The review file contains the listings ID and a date for every review.

### Data Preparation and Modeling

To analyze tweets, the text is cleaned from punctuation and stopwords, stemmed and tokenized.

### Result Evaluation

The web app shows promising results classifying new messages.

## File Descriptions <a name="files"></a>

The results can be viewed in a Flask Web App, which imports the created model to classify new messages and displays visualizations of the training dataset.  


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thanks to insidedairbnb.com for providing latest data on Airbnb listings. The analysis was done with data released in September 2019.
