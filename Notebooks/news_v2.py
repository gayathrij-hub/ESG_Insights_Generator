import requests
import pandas as pd
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
import faiss
import json

class ESGProcessor:
    def __init__(self):
        # ✅ Load SentenceTransformer Model for FAISS Embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ✅ Initialize FAISS Vector Store
        dimension = self.embedding_model.get_sentence_embedding_dimension()  # Get vector size
        self.index = faiss.IndexFlatL2(dimension)  # Create FAISS index
        
        # ✅ Load ESGify Classification Model & Tokenizer
        MODEL_NAME = "ai-lab/ESGify"
        self.classification_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # ✅ Initialize Dictionary to Store FAISS Mappings
        self.stored_perceptions = {}
        
        #newsapi.io
        self.NEWS_API_KEY = os.getenv("NEWS_API_KEY")

    # ✅ Function to Fetch News Data
    def fetch_newsdata(self, company_name, max_results=10):
        """
        Fetches the top news articles related to ESG controversies for a given company using newsdata.io.
        """
        base_url = "https://newsdata.io/api/1/latest?"

        # Construct query to capture ESG controversies (e.g., controversy, emissions, sustainability)
        query = f"{company_name} ESG OR controversy OR emissions OR sustainability"
        
        # Define request parameters
        params = {
            "apikey": self.NEWS_API_KEY,
            "q": query,
            "language": "en",
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            # Convert the results into a DataFrame for easy processing
            articles = data.get("results", [])
            return pd.DataFrame(articles)

            # ✅ Ensure 'description' exists
            if "description" not in df.columns:
                print(f"Warning: 'description' missing for {company_name}. Creating empty column.")
                df["description"] = None  # Create empty column to avoid KeyError
        else:
            print(f"Error fetching news: {response.status_code}")
            return None

    # ✅ Function to Classify ESG Risks
    def classify_news_batch(self, news_df):
        """Classifies each news article's description using the ESGify model."""
        
        if "description" not in news_df.columns:
            print("Skipping classification: No 'description' column in DataFrame.")
            news_df["esg_classification"] = None  # Ensure column exists
            return news_df  

        classifications = [None] * len(news_df)  # Pre-fill with None

        for idx, description in enumerate(news_df["description"]):
            if pd.isna(description):  # Handle missing values
                classifications[idx] = None  # Explicitly assign None
            else:
                inputs = self.tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.classification_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()

                if hasattr(self.classification_model.config, "id2label"):
                    ID2LABEL = self.classification_model.config.id2label  # Use model's label mapping
                else:
                    print("Warning: No ID2LABEL mapping found. Defaulting to generic categories.")
                    ID2LABEL = {i: f"Category {i}" for i in range(len(scores))}

                # ✅ Ensure length consistency
                top_indices = np.argsort(scores)[-3:][::-1]
                top_categories = [(ID2LABEL.get(i, f"Category {i}"), round(scores[i], 3)) for i in top_indices]

                classifications[idx] = top_categories  # Assign to the same index

        # ✅ Store classifications in DataFrame
        news_df["esg_classification"] = classifications
        return news_df

    def compute_esg_perception(self, news_df):
        """
        Aggregates ESG classification scores across all news articles 
        and computes an overall perception score.
        """
        # Extract classifications from DataFrame
        all_risks = []
        all_scores = []

        for classifications in news_df["esg_classification"]:
            if classifications is not None:
                for category, score in classifications:
                    all_risks.append(category)
                    all_scores.append(score)

        if not all_scores:  # Handle empty scores
            return {
                "Average_Perception_Score": None,
                "Dominant_ESG_Risks": [],
                "Risk_Frequency": {}
            }

        # Compute average perception score
        avg_score = np.mean(all_scores)

        # Identify the most common ESG concerns
        risk_counts = Counter(all_risks)
        dominant_risks = risk_counts.most_common(3)  # Top 3 most frequent risks

        # Format results
        esg_summary = {
            "Average_Perception_Score": round(avg_score, 3),
            "Dominant_ESG_Risks": [risk[0] for risk in dominant_risks],
            "Risk_Frequency": dict(risk_counts)
        }

        return esg_summary

    # ✅ Function to Store ESG Perceptions in FAISS
    def store_esg_perception_in_faiss(self, company_name, esg_summary):
        """
        Converts ESG perception summary into embeddings, stores in FAISS, 
        and saves a mapping from FAISS index to actual ESG summary.
        """
        perception_text = f"""
    ESG Perception Summary for {company_name}:
    - Average ESG Risk Score: {esg_summary["Average_Perception_Score"]}
    - Dominant ESG Risks: {", ".join(esg_summary["Dominant_ESG_Risks"])}
    - Risk Breakdown: {esg_summary["Risk_Frequency"]}
    """

        # ✅ Use SentenceTransformer Model to Generate Embedding
        embedding = self.embedding_model.encode([perception_text])

        # ✅ Store Embedding in FAISS
        self.index.add(np.array(embedding))

        # ✅ Store ESG Summary in External Dictionary (Mapped to FAISS)
        self.stored_perceptions[len(self.stored_perceptions)] = perception_text

        return self.index

    # ✅ Function to Retrieve ESG Perceptions from FAISS
    def retrieve_esg_perception(self, query):
        """
        Retrieves the most relevant ESG perception summary from FAISS based on a query.
        """
        query_embedding = self.embedding_model.encode([query])
        _, indices = self.index.search(np.array(query_embedding), k=1)  # Retrieve top match

        retrieved_index = indices[0][0]  # Extract the best match

        if retrieved_index == -1 or retrieved_index not in self.stored_perceptions:
            return "No matching ESG perception found."

        return self.stored_perceptions[retrieved_index]  # Return the actual ESG perception data

    # ✅ Function to Retrieve or Compute ESG Perception
    def retrieve_or_compute_esg_perception(self, company_name):
        """Retrieves ESG perception from FAISS or computes and stores it if missing."""
        # ✅ Step 1: Encode the query and check if the company exists in FAISS
        query_embedding = self.embedding_model.encode([company_name])
        _, indices = self.index.search(np.array(query_embedding), k=1)
        retrieved_index = indices[0][0]  # Extract the best match

        # ✅ Step 2: If the company exists, return the stored perception
        if retrieved_index != -1 and retrieved_index in self.stored_perceptions:
            print(f"Retrieved {company_name}'s ESG perception from FAISS.")
            return self.stored_perceptions[retrieved_index]
        
        print(f"{company_name} not found in FAISS. Computing new ESG perception...")
        news_df = self.fetch_newsdata(company_name)
        news_df = self.classify_news_batch(news_df)
        esg_summary = self.compute_esg_perception(news_df)
        
        return self.store_esg_perception_in_faiss(company_name, esg_summary)

    def populate_vector_store(self, json_file):
        """Populates FAISS vector store with ESG perceptions for first 50 companies."""
        with open(json_file, "r") as file:
            company_data = json.load(file)
        
        first_50_companies = list(company_data.items())[:5]
        
        for _, company in first_50_companies:
            company_name = company["title"]
            print(f"Processing {company_name}...")
            news_df = self.fetch_newsdata(company_name)
            news_df = self.classify_news_batch(news_df)
            esg_summary = self.compute_esg_perception(news_df)
            self.store_esg_perception_in_faiss(company_name, esg_summary)
        
        print("Vector store populated with first 50 companies.")
