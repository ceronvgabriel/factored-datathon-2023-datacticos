import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm

# NLP Pkgs
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BartTokenizer, BartForConditionalGeneration
import openai
openai.api_key = "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"


#Select path to all data
#path_to_master_data='/home/sites/reviews_master_parquet/' # deployment path
path_to_master_data='/home/yhbedoya/Datathon/reviews_master_parquet/' # local path
#path_to_master_data='./master_data/reviews_master_parquet/' # local path Gabriel


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

@st.cache_resource
def generate_brands_list(df):
    print("Preparing brands")
    logging.info(f"Preparing brands")

    brands = df["brand"].unique()
    brands = sorted(brands)
    return brands

#caching data loading so it only happens once, changed cache_data with cache_resource as suggest documentation for large datasets
@st.cache_resource
def load_data_by_category(selected_category):
    print("Loading Data")
    logging.info(f"Loading Data")

    category_file = pd.read_parquet(f'{path_to_master_data}main_cat={selected_category}/', engine="pyarrow")
    print(f"Loaded {selected_category}")
    logging.info(f"Loaded {selected_category}")
    #Add category column, merging Cell Phones & Accessories and Cell Phones &amp; Accessories
    if selected_category == "Cell Phones &amp; Accessories":
        category_file['main_cat'] = "Cell Phones & Accessories"
    else:
        category_file['main_cat'] = selected_category

    print("Finished loading data")
    logging.info(f"Finished loading data")
    print("Finished concatenating data")
    logging.info(f"Finished concatenating data")

    #Filtering dataset
    print("Filtering dataset")
    # Add the length of the review to the dataframe
    print("Adding reviewLen column")
    category_file["reviewLen"] = category_file["reviewText"].apply(lambda x: len(str(x)))
    print("filtering max 3k ch length")
    df = category_file[category_file["reviewLen"]<=3000]
    return df

# Load the pre-trained BART model and tokenizer, and the sentiment analyzer
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
        st.write(f"{i+1} - {summary}")
    return summaries_list

# Generate prompts for the OpenAI GPT-3 model
def generatePrompt(df, selected_product):
    init = f"Between the strings $%& and &%$ there are a number of reviews for the product '{selected_product}'. Generate a list of the main characteristics of the product mentioned in the review, divide it into positives and negatives. Finally generate a set of ideas to improve the product based on your knowledge of simmilar amazon products. display the result as Positive: ... \n Negative: ... \n Improvement oportunities: ...  \n $%& \n"
    reviews = df["reviewText"]
    middle = "\n".join(reviews.tolist())
    end = "\n &%$"

    prompt = init + middle + end
    return prompt

# Generate summaries using the OpenAI GPT-3 model
@st.cache_resource
def getProductInsights(df, selected_product):

    reviews_of_selected_products=selected_products_table[selected_products_table['title']==selected_product]
    reviews_of_selected_products.sort_values("vote", inplace=True, ascending=False)
    df = reviews_of_selected_products.iloc[:max_num_reviews]
    print("! Calling GPT-3 API")

    prompt = generatePrompt(df, selected_product)
    output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user",
              "content":prompt}]
              )    
    insights = output["choices"][0]["message"]["content"]
    st.write(insights)

#Selecting a brand
@st.cache_resource
def select_brand(df_all):
    brands=df_all['brand'].unique() # this query can be avoided by having the brands in a list
    brands.sort()
    brands = list(filter(lambda s: s.strip() != "", brands)) # remove empty strings
    return brands

@st.cache_resource
def generate_summaries(selected_product):
    reviews_of_selected_products=selected_products_table[selected_products_table['title']==selected_product]
    print(f"shape of reviews for product: {reviews_of_selected_products.shape}")

    # Apply the sentiment analysis function to each review
    sentiment_scores = reviews_of_selected_products["reviewText"].apply(get_sentiment_score)

    # Add the sentiment scores as a new column in the DataFrame
    df3 = pd.concat([reviews_of_selected_products, sentiment_scores.rename('sentiment_score')], axis=1)

    # Divide sentiment scores into four bins and count occurrences in each bin
    bins = [-1, 0, 1]
    bin_labels = ['Negative','Positive']
    df3['sentiment_bin'] = pd.cut(df3['sentiment_score'], bins=bins, labels=bin_labels)

    #Sort by number of votes
    df3.sort_values("vote", inplace=True, ascending=False)

    df_neg = df3[df3["sentiment_bin"].isin(['Negative'])].iloc[:max_num_reviews]
    st.write("### Summaries for most voted Negative reviews:",style="color:blue")
    with st.spinner('Generating summaries for most voted Negative reviews...'):
        gen_neg_summaries=generateSummaries(df_neg)
    if not gen_neg_summaries:
        st.write("No Negative reviews found")

    df_pos = df3[df3["sentiment_bin"].isin(['Positive'])].iloc[:max_num_reviews]
    st.write("### Summaries for most voted Positive reviews:",style="color:blue")
    with st.spinner('Generating summaries for most voted Positive reviews...'):
        gen_pos_summaries=generateSummaries(df_pos)
    if not gen_pos_summaries:
        st.write("No Positive reviews found")

#df_all=load_data()
#print("Returned from load_data()")
print("Loading model")
model, tokenizer, analyzer = load_model()
print("Returned from load_model()")

#Max num of reviews to generate
max_num_reviews=5

# Initialize session_state variables
if 'click_summaries' not in st.session_state:
    st.session_state.click_summaries = False

if 'click_insights' not in st.session_state:
    st.session_state.click_insights = False

if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None

if 'selected_brand' not in st.session_state:
    st.session_state.selected_brand = None

if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None


st.write("Select the category of interest")

selected_category = st.selectbox('Select a Category', selected_categories)
if selected_category != st.session_state.selected_category:
    st.session_state.selected_category  = selected_category
    st.session_state.click_summaries = False
    st.session_state.click_insights = False


df_cat = load_data_by_category(selected_category)
print(f"shape of category df: {df_cat.shape}")

st.write("Select a brand and a product to generate summaries of the most voted reviews and product insights")

brandsList = generate_brands_list(df_cat)

selected_brand = st.selectbox('Select a brand', brandsList)
print(f"selected_brand: {selected_brand}")

if selected_brand:
    #"Select a product"
    if selected_brand != st.session_state.selected_brand:
        st.session_state.selected_brand = selected_brand
        st.session_state.click_summaries = False
        st.session_state.click_insights = False

    selected_products_table=df_cat[df_cat["brand"]==selected_brand]
    st.session_state.df_all=selected_products_table
    selected_products=selected_products_table['title'].unique()
    selected_products.sort()
    print(f"Shape of products for brand df {selected_products.shape}")
    selected_product = st.selectbox('Select a product', selected_products)
    if selected_product != st.session_state.selected_product:
        st.session_state.selected_product = selected_product
        st.session_state.click_summaries = False
        st.session_state.click_insights = False

else:
    selected_product = None

st.write("## Generate Product Insights from most voted reviews:")
if st.button("Generate Product Insights") or st.session_state.click_insights:
    #Generate product insights using the OpenAI GPT-3 model
    with st.spinner('Generating product insights...'):
        getProductInsights(selected_products_table, selected_product)
        st.session_state.click_insights = True

st.write("## Generate Product Summaries from most voted reviews:")
if st.button("Generate Product Summaries") or st.session_state.click_summaries:
    generate_summaries(selected_product)
    st.session_state.click_summaries = True