import streamlit as st
import pandas as pd
import numpy as np
import logging
import os

# NLP Pkgs
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BartTokenizer, BartForConditionalGeneration
import openai
openai.api_key = "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"


#Select path to all data
#path_to_master_data='/home/sites/reviews_master_parquet/' # deployment path
path_to_master_data='./master_data/reviews_master_parquet/' # local path


#Select desired categories or read all categories in the folder
#selected_categories=['All Electronics',"Computers","Cell Phones & Accessories","Cell Phones &amp; Accessories"]

# needed to read the main_cat value, can be removed if the column is already in the dataset

selected_categories = os.listdir(path_to_master_data) 
selected_categories.remove('._SUCCESS.crc')
selected_categories.remove('_SUCCESS')
selected_categories=list(map(lambda x: x[9:],selected_categories)) # remove main_cat= prefix from folders


#Create file to log and append logs to it, logging for debugging purposes in deployment

logging.basicConfig(filename='app.log', filemode='w')
logging.info(f"App Started")

#Start app
st.title('Team Datacticos - Amazon Product Reviews')


#caching data loading so it only happens once, changed cache_data with cache_resource as suggest documentation for large datasets
@st.cache_resource
def load_data():
    print("Loading Data")
    logging.info(f"Loading Data")
    df_all=[]
    for category in selected_categories:
        print(f"Loading {category}")
        logging.info(f"Loading {category}")
        category_file = pd.read_parquet(f'{path_to_master_data}main_cat={category}/', engine="pyarrow")
        print(f"Loaded {category}")
        logging.info(f"Loaded {category}")
        #Add category column, merging Cell Phones & Accessories and Cell Phones &amp; Accessories
        if category == "Cell Phones &amp; Accessories":
            category_file['main_cat'] = "Cell Phones & Accessories"
        else:
            category_file['main_cat'] = category

        df_all.append(category_file)
    print("Finished loading data")
    logging.info(f"Finished loading data")
    print("Concatenating data"  )
    logging.info(f"Concatenating data")
    df_concat=pd.concat(df_all)
    print("Finished concatenating data")
    logging.info(f"Finished concatenating data")
    return df_concat

# Load the pre-trained BART model and tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer, analyzer

# Function to calculate sentiment scores for each review
def get_sentiment_score(text):
    text = str(text)
    return analyzer.polarity_scores(text)['compound']

# Generate summaries using the BART model
def generateSummaries(df):
    summaries_list = []
    for i in range(df.shape[0]):
        review = df.iloc[i]["reviewText"]
        # Tokenize the review and perform the summarization
        inputs = tokenizer.encode("summarize: " + review, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=70, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode and print the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries_list.append(summary)
    return summaries_list

# Generate prompts for the OpenAI GPT-3 model
def generatePrompt(df):
    init = "Between the strings $%& and &%$ there are a number of product reviews. Generate a summary of the reviews, divide it into positives and negatives \n $%& \n"
    reviews = df["reviewText"]
    middle = "\n".join(reviews.tolist())
    end = "\n &%$"

    prompt = init + middle + end
    return prompt

# Generate summaries using the OpenAI GPT-3 model
def getProductInsights(df):
    prompt = generatePrompt(df)
    output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user",
              "content":prompt}]
              )    
    return output["choices"][0]["message"]["content"]

df_all=load_data()
print("Returned from load_data()")
print("Loading model")
model, tokenizer, analyzer = load_model()
print("Returned from load_model()")

#Track loaded data in session state to be used in other pages
st.session_state.df_all=df_all


print("Displaying App")
logging.info(f"Returned from load_data(), displaying App")

"Showing loaded data"
st.write("Select a brand and a product to generate summaries of the most voted reviews and product insights")

#Filtering the dataset
@st.cache_resource
def filter_dataset(df):
    # Add the length of the review to the dataframe
    print("Adding reviewLen column")
    df["reviewLen"] = df["reviewText"].apply(lambda x: len(str(x)))
    print("filtering max 3k ch length")
    df = df[df["reviewLen"]<=3000]
    return df


df_all=filter_dataset(df_all)

#Selecting a brand
@st.cache_resource
def select_brand(df_all):
    brands=df_all['brand'].unique() # this query can be avoided by having the brands in a list
    brands.sort()
    brands = list(filter(lambda s: s.strip() != "", brands)) # remove empty strings
    return brands

brands=select_brand(df_all)

selected_brand = st.selectbox('Select a brand', brands)
if selected_brand:
    #"Select a product"
    selected_products_table=df_all[df_all['brand']==selected_brand]
    selected_products=selected_products_table['title'].unique()
    selected_products.sort()
    selected_product = st.selectbox('Select a product', selected_products)
else:
    selected_product = None

@st.cache_resource
def generate_summaries(selected_product):
    print("Running once")
    reviews_of_selected_products=selected_products_table[selected_products_table['title']==selected_product]

    # Apply the sentiment analysis function to each review
    sentiment_scores = reviews_of_selected_products["reviewText"].apply(get_sentiment_score)

    # Add the sentiment scores as a new column in the DataFrame
    df3 = pd.concat([reviews_of_selected_products, sentiment_scores.rename('sentiment_score')], axis=1)

    # Divide sentiment scores into four bins and count occurrences in each bin
    bins = [-1, -0.5, 0, 0.5, 1]
    bin_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive']
    df3['sentiment_bin'] = pd.cut(df3['sentiment_score'], bins=bins, labels=bin_labels)

    #Sort by number of votes
    df3.sort_values("vote", inplace=True, ascending=False)

    df_neg = df3[df3["sentiment_bin"].isin(['Negative', 'Very Negative'])].iloc[:5]
    with st.spinner('Generating summaries for most voted Negative reviews...'):
        gen_neg_summaries=generateSummaries(df_neg)
    st.write("Summaries for most voted Negative reviews:")
    st.write(gen_neg_summaries)

    df_pos = df3[df3["sentiment_bin"].isin(['Positive'])].iloc[:5]

    with st.spinner('Generating summaries for most voted Positive reviews...'):
        gen_pos_summaries=generateSummaries(df_pos)
    st.write("Summaries for most voted Positive reviews:")
    st.write(gen_pos_summaries)

generate_summaries(selected_product)

if st.button("Generate Product Insights"):
    print("! Calling GPT-3 API")

    # #Generate product insights using the OpenAI GPT-3 model
    # with st.spinner('Generating product insights...'):
    #     insights=getProductInsights(df3.iloc[:5])
    # st.write("Product insights:")
    # st.write(insights)

