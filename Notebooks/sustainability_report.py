import requests
import pdfplumber
from io import BytesIO
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os
from googlesearch import search

class ESGRAGRetriever:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", faiss_index_path="sustainability_faiss.index"):
        self.model = SentenceTransformer(embedding_model)
        self.faiss_index_path = faiss_index_path
        self.text_chunks = []
        self.index = None
        self._load_faiss_index()

    def find_sustainability_report(self, prompt):
        """Google search for the company's latest sustainability report."""
        query = f"{prompt} + sustainability report filetype:pdf"
        results = list(search(query, num_results=5))
        return results  # Returns top results, preferably PDF links

    def is_pdf(self, url):
        """Check if the given URL is a PDF file."""
        try:
            headers = requests.head(url).headers
            return "application/pdf" in headers.get("Content-Type", "")
        except Exception as e:
            print(f"Error checking file type: {e}")
            return False

    def extract_text_from_pdf_url(self, pdf_url):
        """Extract text directly from an online PDF."""
        try:
            response = requests.get(pdf_url)
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_html(self, url):
        """Scrape text from an HTML sustainability report."""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text()
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            return ""

    def extract_sustainability_report(self, url):
        """Decides whether to process a PDF or an HTML page."""
        if self.is_pdf(url):
            print("Extracting text from PDF...")
            return self.extract_text_from_pdf_url(url)
        else:
            print("Extracting text from HTML...")
            return self.extract_text_from_html(url)

    def chunk_text(self, text, chunk_size=500):
        """Split text into smaller chunks for efficient retrieval."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def _load_faiss_index(self):
        """Load FAISS index from file and ensure text chunks are reloaded."""
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)
            print(f"✅ FAISS index loaded from {self.faiss_index_path}. Index size: {self.index.ntotal}")

            # Load text chunks (Must be stored separately in a JSON or text file)
            chunk_path = self.faiss_index_path.replace(".index", "_chunks.npy")
            if os.path.exists(chunk_path):
                self.text_chunks = np.load(chunk_path, allow_pickle=True).tolist()
                print(f"✅ Loaded {len(self.text_chunks)} stored text chunks.")
            else:
                print("⚠️ Warning: No text chunks found! FAISS vectors exist but text is missing.")

        else:
            self.index = faiss.IndexFlatL2(384)  # Ensure embedding dimension matches
            print("⚠️ Initialized new FAISS index (No existing index found).")



    def store_embeddings(self, text_chunks):
        """Store new sustainability report embeddings in FAISS, preserving old data."""
        
        # Ensure text chunks are saved along with FAISS embeddings
        self.text_chunks.extend(text_chunks)
        
        embeddings = self.model.encode(text_chunks)
        dimension = embeddings.shape[1]

        # Load existing FAISS index if it exists
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)
            print(f"ℹ️ FAISS index found. Current size: {self.index.ntotal}")
        else:
            self.index = faiss.IndexFlatL2(dimension)
            print("⚠️ No FAISS index found. Initializing a new one.")

        # Add new data to FAISS
        self.index.add(np.array(embeddings))
        
        # Save FAISS index and text chunks
        faiss.write_index(self.index, self.faiss_index_path)
        chunk_path = self.faiss_index_path.replace(".index", "_chunks.npy")
        np.save(chunk_path, np.array(self.text_chunks))

        print(f"✅ Stored {len(text_chunks)} new text chunks. Total chunks: {len(self.text_chunks)}")


    def _fetch_and_store_report(self, prompt):
        """Fetches a new sustainability report, stores it, and re-runs query automatically."""
        report_links = self.find_sustainability_report(prompt)
        if report_links:
            report_url = report_links[0]
            print(f"✅ Processing report from: {report_url}")

            # Extract and store the sustainability report
            report_text = self.extract_sustainability_report(report_url)
            text_chunks = self.chunk_text(report_text)
            self.store_embeddings(text_chunks)

            # Reload FAISS index and text chunks to ensure they are updated
            chunk_path = self.faiss_index_path.replace(".index", "_chunks.npy")
            if os.path.exists(chunk_path):
                self.text_chunks = np.load(chunk_path, allow_pickle=True).tolist()
                print(f"✅ Reloaded {len(self.text_chunks)} text chunks after storing new report.")

            print("✅ New sustainability report stored in FAISS. Now answering your query...")

            # 🔹 Instead of asking the user to retry, re-run query_rag automatically
            return self.query_rag(prompt, retrying=True)

        else:
            return "❌ No sustainability reports found for your query."



    def _generate_gpt_response(self, prompt, context):
        """Calls GPT-4 to generate a response based on retrieved FAISS data."""
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=500,
            messages=[{"role": "system", "content": "You are an ESG expert."},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuery: {prompt}"}]
        )
        return response.choices[0].message.content

    
    def query_rag(self, prompt, threshold=0.60, retrying=False):
        """Retrieve sustainability insights using FAISS & GPT-4.
        
        - Uses FAISS to check if the prompt matches stored data.
        - If similarity is high, retrieves insights.
        - If similarity is low, fetches a new sustainability report and re-runs query.
        """

        # Step 1: Check if FAISS is empty
        if self.index.ntotal == 0:
            print("⚠️ FAISS is empty. Fetching sustainability report...")
            return self._fetch_and_store_report(prompt)

        # Step 2: Compare the prompt with stored FAISS embeddings
        query_embedding = self.model.encode([prompt])
        distances, indices = self.index.search(np.array(query_embedding), k=1)

        # Debug: Check if FAISS returned any results
        if len(indices[0]) == 0 or indices[0][0] == -1:
            print("❌ No relevant matches found in FAISS. Fetching new report...")
            return self._fetch_and_store_report(prompt)

        # Convert FAISS L2 distance to cosine similarity
        similarity = 1 - (distances[0][0] / 2)  # FAISS returns squared L2 distance

        print(f"🔍 Similarity score with stored data: {similarity:.2f}")

        if similarity >= threshold:
            print("✅ High similarity! Using stored sustainability report.")

            # Step 3: Ensure retrieved indices are valid
            valid_indices = [i for i in indices[0] if i < len(self.text_chunks)]
            if not valid_indices:
                print("❌ Retrieved indices are out of range. Fetching new report...")
                if retrying:  # Prevent infinite re-fetching
                    return "❌ Error: Report fetched but indices still invalid. Try again later."
                return self._fetch_and_store_report(prompt)

            # Retrieve the most relevant chunks
            retrieved_docs = [self.text_chunks[i] for i in valid_indices]
            context = "\n".join(retrieved_docs)

            return self._generate_gpt_response(prompt, context)
        else:
            print("⚠️ Low similarity! Fetching a new sustainability report and re-running query...")
            if retrying:  # Prevent infinite re-fetching
                return "❌ Error: Report fetched but similarity still low. Try again later."
            return self._fetch_and_store_report(prompt)

