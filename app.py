# streamlit_app.py

import os
import streamlit as st
import pinecone
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Login to HuggingFace
if huggingface_api_key:
    login(huggingface_api_key)
else:
    st.error("HUGGINGFACE_TOKEN not set in environment")
    st.stop()

# Connect to Pinecone
if not pinecone_api_key:
    st.error("PINECONE_API_KEY not set in environment")
    st.stop()

pc = pinecone.Pinecone(api_key=pinecone_api_key)
index_name = "sql"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# HuggingFace Inference Client
client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=huggingface_api_key)

# Load embedding model
embed_model = SentenceTransformer('google/flan-t5-large')


# Streamlit UI
st.set_page_config(page_title="SQL Generator Bot")
st.title("SQL Generator Bot")
st.markdown("Ask me to create SQL queries!")

user_input = st.text_input("Enter your request", placeholder="e.g., Create a table with columns...")

if st.button("Generate SQL") and user_input:
    with st.spinner("Generating SQL..."):

        query_embedding = embed_model.encode(user_input).tolist()

        try:
            results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            context = ""

            if results.matches:
                for match in results.matches:
                    metadata = match.metadata
                    table_name = metadata.get("table_name", "")
                    columns = metadata.get("columns", "")
                    context += f"Table: {table_name}\nColumns: {columns}\n\n"
        except Exception as e:
            st.error(f"Error querying Pinecone: {e}")
            context = ""

        prompt = f"""You are a helpful SQL assistant.
Given the context:\n{context}
And the following user request:\n{user_input}
Write a full SQL query for this request.
Respond with only the SQL query.
"""

        try:
            response = client.text_generation(
                prompt=prompt,
                max_new_tokens=300,
                temperature=0.2,
                top_p=0.95,
                repetition_penalty=1.1,
            )
            st.success("SQL Generated:")
            st.code(response.strip(), language='sql')
        except Exception as e:
            st.error(f"Error generating SQL: {e}")
