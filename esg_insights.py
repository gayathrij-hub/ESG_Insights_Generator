# esg_insights.py

from pipeline.news import ESGNewsProcessor
from pipeline.sustainability_report import ESGRAGRetriever
import openai
import os
import anthropic

# Replace the following placeholder with the actual Anthropic API call if available.
def query_anthropic(prompt, context):
    """
    Query the Anthropic endpoint (e.g., Claude) for ESG insights using Anthropic's API.
    
    This function builds a prompt that includes the provided context and query,
    then sends it to Anthropic's API using the configured model.
    """
    # Load your API key from an environment variable or directly
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Instantiate the Anthropic client
    client = anthropic.Anthropic(api_key=API_KEY)
    
    # Construct the prompt using Anthropic's formatting tokens
    anthropic_prompt = (
        f"{anthropic.HUMAN_PROMPT}Context:\n{context}\n\nQuery:\n{prompt}{anthropic.AI_PROMPT}"
    )
    
    # Make the API call. Adjust parameters like model name, max tokens, and temperature as needed.
    response = client.completions(
        prompt=anthropic_prompt,
        model="claude-v1",         # You might choose a different variant if needed.
        max_tokens_to_sample=500,
        temperature=0.5,
    )
    
    # Return the generated completion from Anthropic's response.
    return response["completion"]

def query_chatgpt(prompt, context):
    """
    Query ChatGPT (e.g., GPT-4) for ESG insights using OpenAI's API.
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        max_tokens=500,
        messages=[{"role": "system", "content": "You are an ESG expert."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuery: {prompt}"}]
    )
    return response.choices[0].message.content

def main_llm_endpoint(prompt, context, model_choice):
    """
    Route the final LLM call based on the user's model choice.
    
    Args:
        prompt (str): The initial query or company name.
        context (str): Combined context from both ESG pipelines.
        model_choice (str): "ChatGPT" or "Anthropic".
        
    Returns:
        str: The final synthesized ESG insights.
    """
    if model_choice == "ChatGPT":
        return query_chatgpt(prompt, context)
    elif model_choice == "Anthropic":
        return query_anthropic(prompt, context)
    else:
        raise ValueError("Invalid model selection. Please choose either 'ChatGPT' or 'Anthropic'.")

def get_esg_insights(company_name, model_choice="ChatGPT"):
    """
    Retrieves ESG insights by combining news-based ESG perceptions with sustainability report insights,
    and then synthesizes a final output using the chosen LLM endpoint.
    
    Args:
        company_name (str): The target company name.
        model_choice (str): The LLM endpoint to use ("ChatGPT" or "Anthropic").
        
    Returns:
        str: The final ESG insights.
    """
    # Instantiate the pipelines
    esg_processor = ESGNewsProcessor()
    sustainability_retriever = ESGRAGRetriever()
    
    # Retrieve insights from the news-based pipeline
    news_insight = esg_processor.retrieve_or_compute_esg_perception(company_name)
    
    # Retrieve insights from the sustainability report pipeline
    sustainability_insight = sustainability_retriever.query_rag(company_name)
    
    # Combine the outputs from both pipelines into a unified context
    combined_context = (
        f"News-Based ESG Insights:\n{news_insight}\n\n"
        f"Sustainability Report Insights:\n{sustainability_insight}"
    )
    
    # Use the selected LLM endpoint to synthesize a final ESG insight
    final_response = main_llm_endpoint(company_name, combined_context, model_choice)
    
    return final_response

if __name__ == "__main__":
    # For testing purposes, allow execution as a standalone script.
    company_name = input("Enter the company name: ")
    model_choice = input("Choose LLM endpoint (ChatGPT/Anthropic): ")
    insights = get_esg_insights(company_name, model_choice)
    print("\nFinal ESG Insights:")
    print(insights)
