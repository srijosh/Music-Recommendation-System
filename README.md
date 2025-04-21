# 🎵 Music Recommendation System

This project is a content-based music recommender that analyzes the textual features of songs and recommends the top 5 similar ones using TF-IDF and cosine similarity.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Tools and Technologies](#tools-and-technologies)

## Introduction

This system recommends songs based on lyrics and song descriptions, using NLP techniques to compute similarity. Users can choose a song from a dropdown, and the app will display the top 5 most similar songs using precomputed TF-IDF vectors.

## Dataset

- Dataset Source: [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset/data)

## Features

- Computes cosine similarity using TF-IDF

- Interactive UI with dropdown song selector (st.selectbox)

- Displays top 5 similar songs instantly

- WordCloud visualizations and more

## Installation

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Music-Recommendation-System.git
```

2. Navigate to the project directory:

```
   cd Music-Recommendation-System
```

3. Install the required dependencies:

```
   pip install -r requirements.txt
```

## Usage

1. Preprocess the Data:

```
    cd src
    python preprocess.py
```

2. Run the App

```
    streamlit run app.py
```

## Tools and Technologies

- Python

- Streamlit

- Scikit-learn (TF-IDF, cosine similarity)

- NLTK (tokenization, stopwords)

- WordCloud

- Pandas & Matplotlib
