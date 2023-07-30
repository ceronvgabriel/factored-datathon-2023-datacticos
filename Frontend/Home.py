#Select path to all data
path_to_master_data='master_data/parquet/parquet/'
#Select desired categories
selected_categories=['All Electronics',"Computers","Cell Phones & Accessories","Cell Phones &amp; Accessories"]


import streamlit as st
import pandas as pd
import numpy as np


st.title('Team Datacticos - Amazon Product Reviews')

#category_file = pd.read_parquet("master_data/parquet/parquet/main_cat=Computers/", engine="pyarrow")

#caching data loading so it only happens once
@st.cache_data
def load_data():
    print("Loading Data")
    df_all=[]
    for category in selected_categories:

        category_file = pd.read_parquet(f'{path_to_master_data}main_cat={category}/', engine="pyarrow")

        #Add category column merging Cell Phones & Accessories and Cell Phones &amp; Accessories
        if category == "Cell Phones &amp; Accessories":
            category_file['main_cat'] = "Cell Phones & Accessories"
        else:
            category_file['main_cat'] = category

        df_all.append(category_file)

    return pd.concat(df_all)


df_all=load_data()
"Showing loaded data"
st.write(df_all.head(5))

#The following could be optimized by storing the queries for subsequent use

"Show all unique categories sorted alphabetically"
categories=df_all['main_cat'].unique()
categories.sort()
st.write(categories)

selected_category = st.selectbox('Select a category', categories)

"Select a brand"
brands=df_all[df_all['main_cat']==selected_category]['brand'].unique()
brands.sort()
selected_brand = st.selectbox('Select a brand', brands)

"Select a product"
products=df_all[(df_all['main_cat']==selected_category) & (df_all['brand']==selected_brand)]['title'].unique()
products.sort()
selected_product = st.selectbox('Select a product', products)

"Show all reviews for selected product ordered by number of votes"

reviews_by_product = df_all[(df_all['main_cat']==selected_category) & (df_all['brand']==selected_brand) & (df_all['title']==selected_product)]

#astype(int) and replace "NonType" with 0

reviews_by_product["vote"].fillna(0, inplace=True)

reviews_by_product["vote"]=reviews_by_product["vote"].astype(int)

p_reviews_ordered=reviews_by_product.sort_values('vote', ascending=False)
p_reviews_ordered

reviews_list=p_reviews_ordered["reviewText"].to_list()

# Select from list maz reviews
"Select from list maz reviews"

max_number = 10 # just to test, can be changed

if len(reviews_list) > max_number:
    max_reviews = reviews_list[:max_number]
else:
    max_reviews = reviews_list

max_reviews