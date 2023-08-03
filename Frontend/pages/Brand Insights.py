import streamlit as st
import pandas as pd

st.title("Brand Insights")

if "df_all" not in st.session_state:
    st.write("Loading Data")
else:
    df_all=st.session_state.df_all

    #"Checking loaded data"
    #st.write(df_all.head(5))

    #Selecting the brand
    brands=df_all['brand'].unique()
    brands.sort()
    selected_brand = st.selectbox('Select your brand', brands)
    if selected_brand:
        df_brand = df_all[df_all['brand']==selected_brand]
        "Brand main categories in Amazon:"
        count_categories=df_brand['main_cat'].value_counts()
        st.bar_chart(count_categories)

        "Brand reviews activity in time:"
        df_brand['reviewTime'] = pd.to_datetime(df_brand['reviewTime'])
        df_brand['year'] = df_brand['reviewTime'].dt.year
        df_brand['month'] = df_brand['reviewTime'].dt.month
        df_brand['day'] = df_brand['reviewTime'].dt.day

        # Group by year and count number of reviews and plot

        df_brand_year = df_brand.groupby(['year']).count()
        st.line_chart(df_brand_year['reviewerID'])

