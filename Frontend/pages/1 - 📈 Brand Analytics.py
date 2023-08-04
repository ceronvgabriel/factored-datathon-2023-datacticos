import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import altair as alt
st.title("Brand Analytics")

if "df_all" not in st.session_state:
    st.write("Loading Data")
else:
    df_all=st.session_state.df_all

    #Selecting the brand
    brands=df_all['brand'].unique()
    brands.sort()
    selected_brand = st.selectbox('Select your brand', brands)
    if selected_brand:
        df_brand = df_all[df_all['brand']==selected_brand]

        st.write("## Brand main categories in Amazon:")
        count_categories=df_brand['main_cat'].value_counts()
        st.bar_chart(count_categories)

        st.write("## Brand reviews activity in time:")
        df_brand['reviewTime'] = pd.to_datetime(df_brand['reviewTime'])
        df_brand['year'] = df_brand['reviewTime'].dt.year
        df_brand['month'] = df_brand['reviewTime'].dt.month
        df_brand['day'] = df_brand['reviewTime'].dt.day
        # Group by year and count number of reviews and plot
        df_brand_year = df_brand.groupby(['year']).count()
        fig=px.line(df_brand_year["reviewerID"], title='Brand reviews activity in time')
        # alt_fig=alt.Chart(df_brand_year["reviewerID"]).mark_line().encode(
        #     x='year',
        #     y='Count Reviews',
        #     tooltip=['year', 'Count Reviews']
        # )
        #st.altair_chart(alt_fig)
        st.plotly_chart(fig)
        st.line_chart(df_brand_year['reviewerID'],x_axis_label='Year',y_axis_label='Number of reviews')

        st.divider()
        st.write("## Product Popoularity")
        "Which products are the most and least reviewed?"
        
        #Slider to select the year
        years=df_brand['year'].unique()
        if len(years)>1:
            years.sort()
            selected_brand_year = st.slider('Select the year', min_value=years.min(), max_value=years.max(), value=years.max())
        else:
            selected_brand_year=years[0]
        df_brand_year=df_brand[df_brand['year']==selected_brand_year]
        product_count=df_brand_year.groupby(['asin',"title"]).count()
        # sort by coun reviewerID because it has not empty values
        product_count.sort_values("reviewerID", inplace=True, ascending=False)
        product_count=product_count.reset_index()
        product_count_df=product_count[["asin","title","reviewerID"]]
        product_count_df.columns=["ID","Product","Reviews/Sells"]
        #Split product count in half
        product_count_top=product_count_df.iloc[:int(len(product_count_df)/2)]
        product_count_worst=product_count_df.iloc[int(len(product_count_df)/2):]
        st.write("### :green[Popular products:]")
        st.table(product_count_top[:5])
        st.write("### :red[Unpopular Products:]")
        if not product_count_worst.empty:
            
            st.table(product_count_worst[-5:])
        else:
            st.write("No data to display")
        
        # #Commented while we get the overall rating
        #st.divider()

        # st.write("## Product Performance KPI")

        # if len(years)>1:
        #     selected_brand_year_2 = st.slider('Select the year', min_value=years.min(), max_value=years.max(), value=years.max())
        # else:
        #     selected_brand_year_2=years[0]

        # df_brand_year_2=df_brand[df_brand['year']==selected_brand_year_2]
        # st.write(df_brand_year_2.head(5))
        # #Filter by overall rating
        # bad_products=df_brand_year_2[df_brand_year_2['overall']<3]
        # good_products=df_brand_year_2[df_brand_year_2['overall']>=3]
        # #Group by product and count number of reviews
        # bad_products_count=bad_products.groupby(['asin',"title"]).count()
        # good_products_count=good_products.groupby(['asin',"title"]).count()
        # # sort by coun reviewerID because it has not empty values
        # bad_products_count.sort_values("reviewerID", inplace=True, ascending=False)
        # good_products_count.sort_values("reviewerID", inplace=True, ascending=False)
        # bad_products_count=bad_products_count.reset_index()
        # good_products_count=good_products_count.reset_index()
        # bad_products_count_df=bad_products_count[["asin","title","reviewerID"]]
        # good_products_count_df=good_products_count[["asin","title","reviewerID"]]
        # bad_products_count_df.columns=["ID","Product","Reviews"]
        # good_products_count_df.columns=["ID","Product","Reviews"]
        # if not good_products_count_df.empty:
        #     st.write("### :green[Best products:]")
        # else:
        #     st.write("No data to display")
        # st.table(good_products_count_df[:5])
        # st.write("### :red[Products that need attention:]")
        # if not bad_products_count_df.empty:    
        #     st.table(bad_products_count_df[-5:])
        # else:
        #     st.write("No data to display")
        
        # st.divider()

        # st.write("## Customer Satisfaction KPI")
        # "Customer satisfaction is a measure of how well a product or service meets or exceeds the expectations of customers. This KPI is measured by the percentage of customers who are satisfied with a company’s products or services."
        # #Filter by overall rating
        # bad_products=df_brand[df_brand['overall']<3]
        # good_products=df_brand[df_brand['overall']>=3]
        # #Calculate the percentage of bad reviews
        # bad_percentage=bad_products.shape[0]/(bad_products.shape[0]+good_products.shape[0])
        # st.write("### :red[Bad reviews percentage:]")
        # st.write(f"{bad_percentage*100:.2f}%")
        # st.write("### :green[Good reviews percentage:]")
        # st.write(f"{1-bad_percentage*100:.2f}%")

        st.divider()

        st.write("## Top Reviewers")


        count_reviewers=df_brand['reviewerID'].value_counts()
        count_reviewers=count_reviewers.reset_index()
        count_reviewers.sort_values("count", inplace=True, ascending=False)
        count_reviewers.reset_index()
        st.write("### :green[Top Reviewers:]")
        st.table(count_reviewers[:5])

        st.divider()

        st.write("## Customer Lifetime Value KPI")
        "Measures the value of the purchases made by a client"
        # Select reviewer
        reviewers=df_brand['reviewerID'].unique()
        reviewers.sort()
        selected_reviewer = st.selectbox('Select your reviewer', reviewers)
        df_brand_reviewer = df_brand[df_brand['reviewerID']==selected_reviewer]
        
        #Calculate the average purchase value per year
        df_sum_per_year=df_brand_reviewer[['year','price']].groupby("year").sum()
        #df_sum_per_year["year"].astype(str)


        fig = px.line(df_sum_per_year, title='Purcahse value per year')
        st.plotly_chart(fig)
        st.line_chart(df_sum_per_year)
        


        st.divider()

        # Wordcloud of product reviews:
        st.write("## Wordcloud of product reviews:")
        # Select a product:
        products=df_brand['title'].unique()
        products.sort()
        selected_product = st.selectbox('Select your product', products)
        df_brand_product = df_brand[df_brand['title']==selected_product]
        # Create a wordcloud of the product reviews:
        
        @st.cache_data
        def gen_wordcloud(df_brand_product):
            wordcloud = WordCloud(width = 800, height = 800, 
                                background_color ='white', 
                                stopwords = STOPWORDS, 
                                min_font_size = 10).generate(" ".join(df_brand_product['reviewText'].astype(str)))
            return wordcloud
        wordcloud=gen_wordcloud(df_brand_product)
        st.image(wordcloud.to_array())
        


