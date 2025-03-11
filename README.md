# ESG Insights Generator
A Streamlit-powered AI tool to analyze ESG (Environmental, Social, and Governance) trends for companies using sustainability reports and news sentiment analysis.

## Overview

The ESG Insights Generator provides automated sustainability analysis for any company. By leveraging LLMs (Anthropic Claude, ChatGPT) and text analysis techniques, this tool extracts ESG risks, governance transparency, and sustainability recommendations from publicly available reports.

### Key Features:
* ESG Risk Assessment – Extracts environmental, social, and governance risks.
* Sustainability Trends – Analyzes sustainability reports and media sentiment.
* Company-Specific Analysis – Supports querying for any company.
* AI-Powered LLM Integration – Uses Anthropic Claude and ChatGPT for responses.
* Interactive UI – Built with Streamlit for a smooth user experience.

## Installation & Set Up
1. Clone the Repository
 
  ```git clone https://github.com/your-username/ESG_Insights_Generator.git```

  ```cd ESG_Insights_Generator```

2. Create a Virtual Environment
  ```
  # For Windows
  python -m venv venv
  venv\Scripts\activate

  # For Mac/Linux
  python3 -m venv venv
  source venv/bin/activate
  ```

3. Install Dependencies
  ```pip install -r requirements.txt```

4. Set up API Keys
   * Anthropic API Key: Set variable as "ANTHROPIC_API_KEY"
   * NewsData.io API key: https://newsdata.io/ (sign up for the free account API key) and set environment variable as "NEWS_API_KEY"
   Note: Set the API keys as environment variables to ensure precaution
   
## Running the Application
1. Streamlit: ```streamlit run esg_streamlit.py``` OR ```python -m streamlit run esg_streamlit.py```
2. Using ESG Insights in CLI (Optional): ```python esg_insights.py```









