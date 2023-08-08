import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import logging

import matplotlib.pyplot as plt
import openai
openai.api_key = "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"

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BartTokenizer, BartForConditionalGeneration

logging.basicConfig(filename='app.log', filemode='w')
logging.info(f"App Started")

#Select path to all data
#path_to_streaming_data='/home/yhbedoya/Datathon/reviews_streaming_data/' # local path
path_to_streaming_data='/home/sites/reviews_streaming_data/' # deployment path

@st.cache_resource
def load_streaming_data():
    print("Loading Data")
    logging.info(f"Loading Data")

    streamingDf = pd.read_parquet(path_to_streaming_data, engine="pyarrow")
    return  streamingDf

# Load the pre-trained BART model and tokenizer, and the sentiment analyzer
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer, analyzer

@st.cache_resource
def generate_brands_list(df):
    print("Preparing brands")
    logging.info(f"Preparing brands")

    brands = df["brand"].unique()
    brands = sorted(brands)
    return brands

def identify_trending_products(catDf):
    title_counts = catDf["title"].value_counts()[:10]
    # Get the unique values and their corresponding counts from the value_counts() result
    unique_values = title_counts.index
    counts = title_counts.values

    trendDf = catDf[catDf["title"].isin(unique_values)][["title", "brand", "overall", "vote"]]

    brand_dict = {}

    for _, row in trendDf.iterrows():
        title = row['title']
        brand = row['brand']
        brand_dict[title] = brand

    trendDf_no_nan = trendDf.dropna(subset=['overall'])
    mean_overall_by_title = trendDf_no_nan.groupby('title')['overall'].mean()
    mean_overall_dict = mean_overall_by_title.to_dict()

    trendDf_no_nan = trendDf.dropna(subset=['vote'])
    sum_vote_by_title = trendDf_no_nan.groupby('title')['vote'].sum()
    sum_vote_dict = sum_vote_by_title.to_dict()


    trendTitlesDf = pd.DataFrame(title_counts).reset_index()
    trendTitlesDf["brand"] = trendTitlesDf["title"].map(brand_dict)
    trendTitlesDf["Overall"] = trendTitlesDf["title"].map(mean_overall_dict)
    trendTitlesDf["interactions"] = trendTitlesDf["title"].map(sum_vote_dict)
    trendTitlesDf["interactions"] = trendTitlesDf["interactions"] + trendTitlesDf["count"]

    trendTitlesDf['interactions'].fillna(trendTitlesDf['count'], inplace=True)

    # Rename columns
    trendTitlesDf.rename(columns={
        'title': 'product',
        'count': 'total reviews',
        'Overall': 'overall',
        'interactions': 'total interactions'
    }, inplace=True)

    trendTitlesDf['total interactions'] = trendTitlesDf['total interactions'].astype(int)

    desired_columns = ["product", "brand", "total reviews", "overall", "total interactions"]
    df = trendTitlesDf[desired_columns]

    return df

def identify_trending_brands(catDf):
    title_counts = catDf["brand"].value_counts()[:10]
    # Get the unique values and their corresponding counts from the value_counts() result
    unique_values = title_counts.index
    counts = title_counts.values

    trendDf = catDf[catDf["brand"].isin(unique_values)][["brand", "overall", "vote"]]

    trendDf_no_nan = trendDf.dropna(subset=['overall'])
    mean_overall_by_title = trendDf_no_nan.groupby('brand')['overall'].mean()
    mean_overall_dict = mean_overall_by_title.to_dict()

    trendDf_no_nan = trendDf.dropna(subset=['vote'])
    sum_vote_by_title = trendDf_no_nan.groupby('brand')['vote'].sum()
    sum_vote_dict = sum_vote_by_title.to_dict()

    trendTitlesDf = pd.DataFrame(title_counts).reset_index()
    trendTitlesDf["Overall"] = trendTitlesDf["brand"].map(mean_overall_dict)
    trendTitlesDf["interactions"] = trendTitlesDf["brand"].map(sum_vote_dict)
    trendTitlesDf["interactions"] = trendTitlesDf["interactions"] + trendTitlesDf["count"]

    trendTitlesDf['interactions'].fillna(trendTitlesDf['count'], inplace=True)

    # Rename columns
    trendTitlesDf.rename(columns={
        'count': 'total reviews',
        'Overall': 'overall',
        'interactions': 'total interactions'
    }, inplace=True)

    trendTitlesDf['total interactions'] = trendTitlesDf['total interactions'].astype(int)

    desired_columns = ["brand", "total reviews", "overall", "total interactions"]
    df = trendTitlesDf[desired_columns]

    return df

# Generate summaries using the BART model
def generateSummaries(df):
    summaries_list = []
    for i in range(df.shape[0]):
        review = df.iloc[i]["reviewText"]
        if len(review.split(" "))<70:
            summaries_list.append(review)
            continue
        # Tokenize the review and perform the summarization
        inputs = tokenizer.encode("summarize: " + review, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=70, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode and print the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries_list.append(summary)
        
    df["Review summary"] = summaries_list
    return df

# Function to calculate sentiment scores for each review
def get_sentiment_score(text):
    text = str(text)
    return analyzer.polarity_scores(text)['compound']

@st.cache_resource
def generate_summaries_section(reviews_of_selected_products):
    print(f"shape of reviews for product: {reviews_of_selected_products.shape}")
    reviews_of_selected_products.sort_values("vote", inplace=True, ascending=False)
    reviews_of_selected_products2 = reviews_of_selected_products.iloc[:5]
    # Apply the sentiment analysis function to each review
    sentiment_scores = reviews_of_selected_products2["reviewText"].apply(get_sentiment_score)

    # Add the sentiment scores as a new column in the DataFrame
    df3 = pd.concat([reviews_of_selected_products2, sentiment_scores.rename('sentiment_score')], axis=1)

    # Divide sentiment scores into four bins and count occurrences in each bin
    bins = [-1, 0, 1]
    bin_labels = ['Negative','Positive']
    df3['sentiment_bin'] = pd.cut(df3['sentiment_score'], bins=bins, labels=bin_labels)
    df3=generateSummaries(df3)

    desired_columns = ["Review summary", "vote", "sentiment_bin"]
    df3 = df3[desired_columns]

    df3["vote"].fillna(0, inplace=True)
    df3['vote'] = df3['vote'].astype(int)

        # Rename columns
    df3.rename(columns={
        'vote': 'votes',
        'sentiment_bin': 'sentiment'
    }, inplace=True)

    return df3


def product_summaries_insights(df):
    df = df[["reviewText", "vote", "brand"]]
    summariesdf = generate_summaries_section(df)
    summariesdf.reset_index(inplace=True, drop=True)
    return summariesdf

def product_sentiment_analysis(productDf):
    # Apply the sentiment analysis function to each review
    sentiment_scores = productDf["reviewText"].apply(get_sentiment_score)

    # Add the sentiment scores as a new column in the DataFrame
    df3 = pd.concat([productDf, sentiment_scores.rename('sentiment_score')], axis=1)

    # Divide sentiment scores into four bins and count occurrences in each bin
    bins = [-1, 0, 1]
    bin_labels = ['Negative','Positive']
    df3['sentiment_bin'] = pd.cut(df3['sentiment_score'], bins=bins, labels=bin_labels)

    count_sentiment = df3['sentiment_bin'].value_counts().to_dict()

    #st.table(count_sentiment)
    return count_sentiment

def plot_horizontal_bar_chart(data_dict):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_dict.items(), columns=["Sentiment", "Count"])

    # Create the horizontal bar plot using plotly.express
    fig = px.bar(
        df,
        x='Count',
        y='Sentiment',
        orientation='h',
        text='Count',
        color='Sentiment',  # Use the Sentiment column for coloring the bars
        color_discrete_map={'Positive': 'green', 'Negative': 'red'}  # Specify custom colors
    )

    # Customize the layout of the plot
    fig.update_layout(
        title="Sentiment Analysis",
        xaxis_title="Count",
        yaxis_title="Sentiment",
        showlegend=False
    )

    # Display the plot
    st.plotly_chart(fig)

def plot_scatter(df):
    # Create the plot using Plotly Express with gradient color scale
    fig = px.scatter(
        df, x='total reviews', y='overall', color='total interactions', text='brand',
        hover_data={'total reviews': True, 'overall': True, 'total interactions': True},
        color_continuous_scale='RdBu'  # Choose a color scale (you can change it if you prefer)
    )

    # Customize the plot
    fig.update_traces(textposition='top center')
    fig.update_layout(
        title='Product Reviews and Interactions',
        xaxis_title='Total Reviews',
        yaxis_title='Overall Rating',
        showlegend=False
    )

    # Display the plot
    st.plotly_chart(fig)

# Generate prompts for the OpenAI GPT-3 model
def generatePrompt(dfText):
    
    init = f"""Make an short interpretation of the data of the table provided between the tags $%& and &%$ in terms of Brand Perception, Engagement, Market Popularity. \n 
    Dont define Brand Perception, Engagement, Market Popularity. Don't mentione the word table \n 
    display the solution as: \n * Brand perception: ... \n * Engagement: ... \n * Market popularity:... \n Finally identify the strongest and weakest brand supporting your election \n $%&"""

    middle = dfText
    end = "\n &%$"

    prompt = init + middle + end
    return prompt

@st.cache_resource
def generatePlotHints(dfText):
    dfText = dfText.to_string()
    prompt = generatePrompt(dfText)
    output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user",
              "content":prompt}]
              )    
    insights = output["choices"][0]["message"]["content"]
    return insights


max_num_reviews= 5
#Load models

streamingDf = load_streaming_data()
categories = streamingDf["main_cat"].unique().tolist()

st.title("Streaming data analysis")
st.write("##### On this page, you'll discover analytics and valuable insights into the latest market activities. Stay up-to-date with what's happening in the market during the last few days. ")

model, tokenizer, analyzer = load_model()

selected_category = st.selectbox('Select a Category', categories)

brandsList = generate_brands_list(streamingDf)

if selected_category:
    categoryDf=streamingDf[streamingDf['main_cat']==selected_category]

    st.divider()

    st.write(f"### Trending products in the :blue[{selected_category}] category")
    st.write(f"This list shows the products with more interactions during the last 7 days.")

    trend_products_by_catDf = identify_trending_products(categoryDf)

    st.table(trend_products_by_catDf)

    st.write(f"### What is the people talking about these products?")
    st.write(f"Select a trending product from the list and discover insightful information about its reviews.")

    selected_product = st.selectbox('Select a product', trend_products_by_catDf["product"].tolist())

    productDf = streamingDf[streamingDf["title"]==selected_product]

    sentimentBins = product_sentiment_analysis(productDf)
    plot_horizontal_bar_chart(sentimentBins)

    st.write(f"Take a look of the most relevant reviews for this product")

    product_summaries_insightsDf = product_summaries_insights(productDf)
    st.table(product_summaries_insightsDf)

    st.divider()

    st.write(f"### Trending brands in the :blue[{selected_category}] category")
    st.write(f"The following chart provides an overview of customer perception and engagement across different branches within the category in the last 7 days. It illustrates how customers perceive each branch and their level of engagement based on their total interactions with the brand.")

    trend_brands_by_catDf = identify_trending_brands(categoryDf)


    #st.table(trend_brands_by_catDf)

    plot_scatter(trend_brands_by_catDf)

    st.write("###### Would you like some hints about the above visualization?")
    if st.button("Generate hints"):
        hints = generatePlotHints(trend_brands_by_catDf)
        st.write(hints)

    #st.divider()

    #st.write(f"### What is the people talking about these brands?")
    #st.write(f"Select a trending brand from the list and discover insightful information about its reviews.")

