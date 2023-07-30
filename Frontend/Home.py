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

        #Add category column, merging Cell Phones & Accessories and Cell Phones &amp; Accessories
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

#"Select a category"
unique_categories=df_all['main_cat'].unique()
unique_categories.sort()
st.write(unique_categories) # Show all unique categories sorted alphabetically
selected_category = st.selectbox('Select a category', unique_categories)

#"Select a brand"
selected_categories_table=df_all[df_all['main_cat']==selected_category]
brands=selected_categories_table['brand'].unique()
brands.sort()
selected_brand = st.selectbox('Select a brand', brands)

#"Select a product"
selected_products_table=selected_categories_table[selected_categories_table['brand']==selected_brand]
selected_products=selected_products_table['title'].unique()
selected_products.sort()
selected_product = st.selectbox('Select a product', selected_products)

reviews_of_selected_products=selected_products_table[selected_products_table['title']==selected_product]

"Show all reviews for selected product ordered by number of votes"
reviews_of_selected_products["vote"].fillna(0, inplace=True) # replace NaN with 0
reviews_of_selected_products["vote"]=reviews_of_selected_products["vote"].astype(int) # Treat vote as integer
p_reviews_ordered=reviews_of_selected_products.sort_values('vote', ascending=False) # Sort by vote
p_reviews_ordered



# Selecting a maximum number of reviews
"Selecting a maximum number of reviews"

max_number = 10 # just to test, can be changed

reviews_list=p_reviews_ordered["reviewText"].to_list()

if len(reviews_list) > max_number:
    max_reviews = reviews_list[:max_number]
else:
    max_reviews = reviews_list

max_reviews