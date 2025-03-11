import streamlit as st
from esg_insights import get_esg_insights  # Import your existing ESG function

# Streamlit UI
st.set_page_config(page_title="ESG Insights Generator", layout="wide")

st.title("🌱 ESG Insights Generator")
st.write(
    """🔎 **Discover ESG Insights & Sustainability Trends**  
    Enter a company name to generate a comprehensive **Environmental, Social, and Governance (ESG) analysis**.  
    This tool leverages AI to provide insights from **sustainability reports, news sentiment, and industry data**,  
    helping you assess a company's ESG performance, risks, and recommendations.  
    """
    )

# User Input
company_name = st.text_input("Enter Company Name", "")
model_choice = st.selectbox("Choose LLM Model", ["Anthropic", "ChatGPT"])

# Run analysis when user clicks button
if st.button("Generate ESG Insights"):
    if company_name:
        st.write("🔍 Fetching ESG insights, please wait...")
        insights = get_esg_insights(company_name, model_choice)
        
        st.subheader("📊 ESG Insights Report")
        st.write(insights)  # Display result
        
    else:
        st.warning("⚠️ Please enter a company name.")

