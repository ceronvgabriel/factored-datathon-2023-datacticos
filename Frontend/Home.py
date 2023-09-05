import streamlit as st
import logging

logging.basicConfig(filename='app.log', filemode='w')
logging.info(f"App Started")


#Start app
st.write('# Team Datacticos')

st.markdown("""

Welcome to Datacticos Review Insight, your go-to application for gaining valuable insights from Amazon product reviews! 📈 \n
Have you ever wondered what customers really think about a product on Amazon? Are you looking to improve your product or understand its strengths and weaknesses better? Our tool assists you in extracting meaningful information from the vast sea of Amazon reviews. Our tool is composed of three key components:\n\n
:star2: **Amazon Review Analyzer**: enables you to delve deep into your product reviews and gain comprehensive feedback. Discover valuable insights about your customers and use this knowledge to enhance your product and elevate customer satisfaction. It provides you:

- **Positive Characteristics**:
Our app will identify the most frequently mentioned positive attributes in the reviews, giving you a clear understanding of what customers love about your product. \n
- **Negative Characteristics**:
We help you pinpoint the areas where your product may fall short. By analyzing negative reviews, you'll gain valuable insights into what aspects need attention.\n
- **Improvement Opportunities**:
Our application will present you with actionable improvement opportunities, directly sourced from customer feedback, helping you enhance your product to exceed customer expectations.\n
- **Summarized Reviews**:
Save time and effort with our summary feature! We highlight the most relevant parts of both positive and negative reviews, so you can quickly grasp the overall sentiment and key points without having to read through each review.\n\n
"""
)
            
link1 = '[Go to Amazon Review Analyzer](https://datacticos-datathon.azurewebsites.net/Amazon_Review_Analyzer)'
st.markdown(link1, unsafe_allow_html=True)

st.markdown("""
            
:star2: **Brand Analytics**:
Tailor-made to offer you an all-encompassing understanding of your brand's performance, drawing insights from customer reviews and interactions. It provides you: \n\n
- **Analysis of the Reviews Activity in Time:**: Gain a clear understanding of how customer engagement fluctuates throughout different periods.\n
- **Overall Rating of Your Products Over Time**: With our tool, you can effortlessly monitor changes in your product's ratings over time. Observe shifts in customer perception and take proactive measures to maintain and improve your brand's reputation. \n
- **Analysis of Customer Satisfaction**: Measures and evaluates how well your products meet or exceed customer expectations.\n
- **Analysis of Product Features Impacting Customer Satisfaction**: Identify the specific product features that play a significant role in driving customer satisfaction.\n
"""
)
            
link2 = '[Go to Brand Analytics](https://datacticos-datathon.azurewebsites.net/Brand_Analytics)'
st.markdown(link2, unsafe_allow_html=True)

st.markdown("""
:star2: **Streaming Analytics**:
Stay updated on real-time market trends, popular products, and emerging brands, empowering you with the latest market intelligence.\n\n

- **Trending Products by Market**: Allows you to identify the hottest products in the market and what the customers are talking about it.\n
- **Trending Brands by Market**: Discover the brands that are making waves in the market during the last days. \n
""")
link3 = '[Go to Streaming Analytics](https://datacticos-datathon.azurewebsites.net/Streaming_Analytics)'
st.markdown(link3, unsafe_allow_html=True)

st.markdown("""
We are committed to empowering businesses with valuable information to make data-driven decisions. Our goal is to help you unlock the full potential of your product and deliver exceptional customer experiences.
""")
