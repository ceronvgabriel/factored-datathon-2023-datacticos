import streamlit as st
import pandas as pd

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
        st.line_chart(df_brand_year['reviewerID'])

        st.divider()
        st.write("## Product Performance KPI")
        "Which products are top-selling, and which are behind the pack? This product performance KPI ranks products based on sales."
        product_count=df_brand.groupby(['asin',"title"]).count()
        # sort by coun reviewerID because it has not empty values
        product_count.sort_values("reviewerID", inplace=True, ascending=False)
        product_count=product_count.reset_index()
        product_count_df=product_count[["asin","title","reviewerID"]]
        product_count_df.columns=["ID","Product","Reviews/Sells"]
        #Split product count in half
        product_count_top=product_count_df.iloc[:int(len(product_count_df)/2)]
        product_count_worst=product_count_df.iloc[int(len(product_count_df)/2):]
        st.write("### :green[Top Products:]")
        st.table(product_count_top[:5])
        st.write("### :red[Products that need attention:]")
        if not product_count_worst.empty:
            
            st.table(product_count_worst[-5:])
        else:
            st.write("No data to display")
        