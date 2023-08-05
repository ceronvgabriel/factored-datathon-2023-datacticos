import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import numpy as np
import os
import logging

logging.basicConfig(filename='app.log', filemode='w')
logging.info(f"App Started")

#Select path to all data
#path_to_master_data='/home/sites/reviews_master_parquet/' # deployment path
#path_to_master_data='/home/yhbedoya/Datathon/reviews_master_parquet/' # local path
path_to_master_data='./master_data/reviews_master_parquet_2/' # local path Gabriel


#Select desired categories or read all categories in the folder
#selected_categories=['All Electronics',"Computers","Cell Phones & Accessories","Cell Phones &amp; Accessories"]

# needed to read the main_cat value, can be removed if the column is already in the dataset

selected_categories = os.listdir(path_to_master_data) 
selected_categories.remove('._SUCCESS.crc')
selected_categories.remove('_SUCCESS')
selected_categories=list(map(lambda x: x[9:],selected_categories)) # remove main_cat= prefix from folders

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

st.title("Brand Analytics")
st.write("#### In this page you can find analytics and insights about your brand and its products, you can use this to take informed decisions and increase your sales, customer satisfaction and brand awareness.")
selected_category = st.selectbox('Select a Category', selected_categories)

df_all = load_data_by_category(selected_category)
print(f"shape of category df: {df_all.shape}")

brandsList = generate_brands_list(df_all)

# #Selecting the brand
# brands=df_all['brand'].unique()
# brands.sort()

selected_brand = st.selectbox('Select your brand', brandsList)
if selected_brand:
    df_brand = df_all[df_all['brand']==selected_brand]
    # #Commenting because we are loading categories dinaamically now, and don't have infor about the main_cat
    # st.write("## Brand main categories in Amazon:")
    # count_categories=df_brand['main_cat'].value_counts()
    # st.bar_chart(count_categories)

    # st.write(df_brand.head(5))
    st.divider()
    # Total products
    st.write(f"## Brand products: :blue[{df_brand['asin'].nunique()}]")
    "Number of products with at least 5 reviews that your brand has in Amazon"
    #Total product reviews
    st.write(f"## Brand total product reviews: :blue[{df_brand.shape[0]}]")
    "Number of reviews for those products"

    st.write("## Brand reviews activity in time:")
    "Here you have a general number of reviews/interactions your customers have had with all your products, by years. It gives clue of your customer engagement and if needs to be improved"
    # Get time from reviewTime
    df_brand['reviewTime'] = pd.to_datetime(df_brand['reviewTime'])
    df_brand['year'] = df_brand['reviewTime'].dt.year
    df_brand['month'] = df_brand['reviewTime'].dt.month
    df_brand['day'] = df_brand['reviewTime'].dt.day
    # Group by year and count number of reviews and plot
    df_brand_year = df_brand.groupby(['year']).count()
    fig=px.line(df_brand_year["reviewerID"], markers=True)
    #Change legend name
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("reviewerID", "Reviews")))
    # Change the name of the axis
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Count Reviews')
    st.plotly_chart(fig)

    # Line chart for overall rating by year
    st.write("## Brand Overall Rating")
    "It gives you a clue of how your customers are rating your products. It can be used to measure the quality of your products, the customer satisfaction by year and if you need to change your general brand strategy. It is calculated with the average rating of all the products"

    df_brand_year=df_brand[["year","overall"]].groupby(['year']).mean()
    fig=px.line(df_brand_year["overall"], markers=True)
    #Change legend name
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("overall", "Overall Rating")))
    # Change the name of the axis
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Overall Rating')
    st.plotly_chart(fig)

    st.write("## Brand Customer Satisfaction")
    "Customer satisfaction is a measure of how well a product meets or exceeds the expectations of customers. This KPI is measured by the percentage of customers who are satisfied (>=3 on review rating) with the brand products."
    #Filter by overall rating
    bad_products=df_brand[df_brand['overall']<3]
    good_products=df_brand[df_brand['overall']>=3]
    #Calculate the percentage of bad reviews
    bad_percentage=bad_products.shape[0]/(bad_products.shape[0]+good_products.shape[0])
    st.write("### :green[Good reviews percentage:]")
    st.write(f"{100-bad_percentage*100:.2f}%")
    st.write("### :red[Bad reviews percentage:]")
    st.write(f"{bad_percentage*100:.2f}%")

    st.divider()

    # st.write("## Product Popoularity")
    # top_number=5
    # f"Which products are the most and least reviewed? (Showing the top and last {top_number} products)"
    # #Slider to select the year
    # years=df_brand['year'].unique()
    # if len(years)>1:
    #     years.sort()
    #     selected_brand_year = st.slider('Select the year', min_value=years.min(), max_value=years.max(), value=years.max(), key="popularity")
    # else:
    #     selected_brand_year=years[0]
    #     st.write(f"Year: {selected_brand_year}")
    # df_brand_year=df_brand[df_brand['year']==selected_brand_year]
    # product_count=df_brand_year.groupby(['asin',"title"]).count()
    # # sort by coun reviewerID because it has not empty values
    # product_count.sort_values("reviewerID", inplace=True, ascending=False)
    # product_count=product_count.reset_index()
    # product_count_df=product_count[["asin","title","reviewerID"]]
    # product_count_df.columns=["ID","Product","Reviews/Sells"]
    # #Split product count in half
    # product_count_top=product_count_df.iloc[:int(len(product_count_df)/2)]
    # product_count_worst=product_count_df.iloc[int(len(product_count_df)/2):]
    # st.write("### :green[Popular products:]")
    # st.table(product_count_top[:5])
    # st.write("### :red[Unpopular Products:]")
    # st.table(product_count_worst[-5:])

    # st.divider()

    st.write("## Product Performance KPI")
    "This KPI measures the performance of your products, you can take decisions as increasing the production and marketing of best products, or remove products with bad performance or improve them."
    years=df_brand['year'].unique()
    # Generate slider only with years with more than 1 review
    if len(years)>1:
        selected_brand_year_2 = st.slider('Select the year', min_value=years.min(), max_value=years.max(), value=years.max())
    else:
        selected_brand_year_2=years[0]
        st.write(f"Year: {selected_brand_year_2}")
    #Filter for selected year
    df_brand_year_2=df_brand[df_brand['year']==selected_brand_year_2]
    #st.write(df_brand_year_2.head(5))
    #Filter by overall rating
    bad_products=df_brand_year_2[df_brand_year_2['overall']<3]
    good_products=df_brand_year_2[df_brand_year_2['overall']>=3]
    good_p_sub=good_products[["asin","title","vote"]]
    good_p_sub=good_p_sub.groupby(['asin',"title"]).sum()
    good_p_sub["count_reviews"]=good_products.groupby(['asin',"title"]).count()["reviewerID"]
    good_p_sub.sort_values("vote", inplace=True, ascending=False)
    good_p_sub=good_p_sub.reset_index()
    good_p_sub.columns=["PID","Product","sum(votes)","count(reviews)"]

    bad_p_sub=bad_products[["asin","title","vote"]]
    bad_p_sub=bad_p_sub.groupby(['asin',"title"]).sum()
    bad_p_sub["count_reviews"]=bad_products.groupby(['asin',"title"]).count()["reviewerID"]
    bad_p_sub.sort_values("vote", inplace=True, ascending=False)
    bad_p_sub=bad_p_sub.reset_index()
    bad_p_sub.columns=["PID","Product","sum(votes)","count(reviews)"]
    st.write("### :green[Best products:]")
    "Best products are the ones with 3 or more overall rating and more votes"
    st.table(good_p_sub[:5])
    st.write("### :red[Products that need atention:]")
    "Not so good products are the ones with less than 3 overall rating and more votes"
    st.table(bad_p_sub[:5])

    st.divider()

    
    
    # st.write("## Top Reviewers")
    # count_reviewers=df_brand['reviewerID'].value_counts()
    # count_reviewers=count_reviewers.reset_index()
    # count_reviewers.sort_values("count", inplace=True, ascending=False)
    # count_reviewers.reset_index()
    # count_reviewers.columns=["ID","Reviews"]
    # "This are your top reviewers, you can give them special benefits as discounts or notifications about new products"
    # st.table(count_reviewers[:5])

    # st.divider()

    # st.write("## Customer Lifetime Value KPI")
    # "Measures the value of the purchases made by a client by year, only shows clients with bought price information"
    # #Drop rows with no price and NaN
    # df_brand_wo_nan=df_brand.copy()
    # df_brand_wo_nan["price"]=pd.to_numeric(df_brand_wo_nan['price'], errors='coerce')
    # df_brand_wo_nan.dropna(subset=['price'], inplace=True)
    # df_brand_w_price=df_brand_wo_nan[df_brand_wo_nan["price"].astype(int)!=0]

    # w_price_bar=df_brand_w_price[["reviewerID","price"]]
    
    # w_price_bar=w_price_bar.groupby("reviewerID").sum()
    # w_price_bar.sort_values("price", inplace=True, ascending=False)
    # w_price_bar=w_price_bar.reset_index()
    # w_price_bar.columns=["ID","Price"]
    
    # fig=px.bar(w_price_bar[:5], x="ID", y="Price", title="Customer Lifetime Value")
    # fig.update_xaxes(title_text='Reviewer ID')
    # fig.update_yaxes(title_text='Purchase Sum Value ($USD)')

    # st.plotly_chart(fig)

    # # Select reviewer, only reviewers which have bought something
    # reviewers=df_brand_w_price['reviewerID'].unique()
    # reviewers.sort()
    
    # selected_reviewer = st.selectbox('Select your reviewer', reviewers)
    # df_brand_reviewer = df_brand_w_price[df_brand_w_price['reviewerID']==selected_reviewer]
    
    # #Calculate the average purchase value per year
    # df_sum_per_year=df_brand_reviewer[['year','price']].groupby("year").sum()
    
    # fig = px.bar(df_sum_per_year, x=df_sum_per_year.index, y="price", title="Customer Lifetime Value")
    # fig.update_xaxes(title_text='Year')
    # fig.update_yaxes(title_text='Purchase Sum Value ($USD)')
    # st.plotly_chart(fig)

    # st.divider()

    # Analytics by product:
    st.write("# Brand products:")
    st.write(f"### Select a product to see analytics and get insights about it; Selected brand: :blue[{selected_brand}]")
    "(To get a deeper understanding of the product reviews refer to the main page Home)"
    # Select a product:
    products=df_brand['title'].unique()
    products.sort()
    selected_product = st.selectbox('Select your product', products)
    df_brand_product = df_brand[df_brand['title']==selected_product]

    st.write(f"### Number of product reviews: {df_brand_product.shape[0]}")

    st.divider()

    st.write("## Product rating in time:")
    "Here you can see the rating of the product in time, it can be used to measure the quality of the product get a better understanding of its customer satisfaction"

    df_brand_product_year=df_brand_product[["year","overall"]].groupby(['year']).mean()
    fig=px.line(df_brand_product_year["overall"], markers=True)
    #Change legend name
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("overall", "Overall Rating")))
    # Change the name of the axis
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Overall Rating')
    st.plotly_chart(fig)

    st.write("## Sentiment Analysis of product reviews:")
    "Here you can have a deeper understanding of the product reviews sentiment"

    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk

    @st.cache_data
    def load_analyzer():
        nltk.download('vader_lexicon')
        analyzer = SentimentIntensityAnalyzer()
        return analyzer
    
    analyzer=load_analyzer()

    # Function to calculate sentiment scores
    def get_sentiment_score(text):
        text = str(text)
        return analyzer.polarity_scores(text)['compound']
    #Created this function just to cache
    @st.cache_resource
    def apply_sentiment_scores(df_brand_product):
        return df_brand_product["reviewText"].apply(get_sentiment_score)
    
    sentiment_scores = apply_sentiment_scores(df_brand_product)
    # Add the sentiment scores as a new column in the DataFrame
    df3 = pd.concat([df_brand_product, sentiment_scores.rename('sentiment_score')], axis=1)
    # Divide sentiment scores into four bins and count occurrences in each bin
    bins = [-1, -0.5, 0, 0.5, 1]
    bin_labels = ['Very Negative', 'Negative', 'Neutral', 'Very Positive']
    df3['sentiment_bin'] = pd.cut(df3['sentiment_score'], bins=bins, labels=bin_labels)


    # Display bar chart of sentiment analysis
    
    fig=px.bar(df3["sentiment_bin"])
    st.plotly_chart(fig)


    # Create a wordcloud of the product reviews:
    st.write("## Product reviews wordcloud:")
    "See the most common words in the reviews of your product, it can be used to get a better understanding of your customer needs and their opinions"
    @st.cache_data
    def gen_wordcloud(df_brand_product):
        wordcloud = WordCloud(width = 800, height = 800, 
                            background_color ='white', 
                            stopwords = STOPWORDS, 
                            min_font_size = 10).generate(" ".join(df_brand_product['reviewText'].astype(str)))
        return wordcloud
    wordcloud=gen_wordcloud(df_brand_product)
    st.image(wordcloud.to_array())
    


