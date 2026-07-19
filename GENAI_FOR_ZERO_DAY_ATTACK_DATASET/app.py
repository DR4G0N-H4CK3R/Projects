import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import requests
import time
import sys

# --- Agentic AI & RAG Imports ---
AGENTIC_AI_AVAILABLE = False
import_error_message = ""

try:
    from langchain_ollama import ChatOllama
    AGENTIC_AI_AVAILABLE = True
except ImportError as e:
    import_error_message = str(e)

# --- API Configuration ---
API_BASE_URL = "http://localhost:8000"

# --- Page Config & Styling ---
st.set_page_config(page_title="Cyber-Twin AI Platform", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2227; padding: 15px; border-radius: 10px; border: 1px solid #3e444e; }
    </style>
    """, unsafe_allow_html=True)

# --- Automated Data Pipeline ---
@st.cache_data
def load_and_preprocess_data(file_buffer):
    if file_buffer.name.endswith('.csv'):
        df_raw = pd.read_csv(file_buffer)
    else:
        df_raw = pd.read_excel(file_buffer)
    df_numeric = df_raw.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_numeric.columns.tolist(), df_raw

# --- Main UI ---
st.title("🛡️ Enterprise Cyber-Twin & Agentic RAG Platform")
st.write("End-to-End Workflow: UI Remote Control -> FastAPI Backend -> LLM Analyst")

# Check if backend is alive
try:
    backend_status = requests.get(f"{API_BASE_URL}/").json()
    st.sidebar.success(f"🟢 Backend Online (Device: {backend_status.get('device', 'unknown').upper()})")
except:
    st.sidebar.error("🔴 Backend Offline (Ensure uvicorn is running on port 8000)")

tab_data, tab_train, tab_evaluate, tab_agent = st.tabs([
    "📁 1. Data Ingestion", "🧠 2. Remote Model Tuning", "📊 3. Digital Twin Output", "🤖 4. Agentic RAG Analyst"
])

if 'synthetic_df' not in st.session_state:
    st.session_state.synthetic_df = None
if 'target_features' not in st.session_state:
    st.session_state.target_features = None

# --- TAB 1: Data Ingestion ---
with tab_data:
    st.subheader("Upload Knowledge Source (Network Baseline)")
    uploaded_file = st.file_uploader("Upload IDS2025 Dataset (CSV/XLSX)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        features, raw_df = load_and_preprocess_data(uploaded_file)
        st.session_state.target_features = features
        st.success(f"✅ Data parsed successfully. Ready to send to backend API.")
        with st.expander("Preview Raw Baseline Data"):
            st.dataframe(raw_df.head())

# --- TAB 2: Remote Model Fine-Tuning ---
with tab_train:
    st.subheader("WGAN-GP Hyperparameter Tuning")
    col1, col2, col3 = st.columns(3)
    with col1:
        LATENT_DIM = st.slider("Latent Noise", 10, 100, 32)
        BATCH_SIZE = st.number_input("Batch Size", value=64)
    with col2:
        HIDDEN_DIM = st.selectbox("LSTM Hidden Units", [32, 64, 128, 256], index=1)
        EPOCHS = st.number_input("Epochs", value=5, min_value=1)
    with col3:
        LAMBDA_GP = st.slider("Gradient Penalty (λ)", 1.0, 20.0, 10.0)
        LR = st.number_input("Learning Rate", value=0.0002, format="%.5f")

    if st.button("🚀 Trigger Remote Training Job"):
        try:
            params = {
                "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM, 
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, 
                "lr": LR, "lambda_gp": LAMBDA_GP
            }
            response = requests.post(f"{API_BASE_URL}/train", params=params)
            
            if response.status_code == 200:
                st.info("API Job Accepted. Monitoring background training...")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                while True:
                    status_res = requests.get(f"{API_BASE_URL}/status").json()
                    if status_res["status"] == "failed":
                        st.error("Training failed on the backend.")
                        break
                        
                    progress = status_res["progress"]
                    progress_bar.progress(float(progress))
                    status_text.text(f"Epoch {status_res['current_epoch']}/{status_res['total_epochs']} | Critic Loss: {status_res['c_loss']:.4f}")
                    
                    if status_res["status"] == "completed":
                        st.success("✅ Backend Training Complete! Proceed to Step 3.")
                        break
                        
                    time.sleep(1)
            else:
                st.error(f"API Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Is FastAPI running?")

# --- TAB 3: Digital Twin Evaluation ---
with tab_evaluate:
    st.subheader("Fetch Synthetic Data from Backend")
    num_samples = st.number_input("Number of synthetic flows to generate", value=1000, min_value=10)
    
    if st.button("📥 Generate & Download"):
        try:
            res = requests.get(f"{API_BASE_URL}/generate", params={"num_samples": num_samples})
            if res.status_code == 200:
                data = res.json()["synthetic_data"]
                data_np = np.array(data)
                flattened_data = data_np.reshape(-1, data_np.shape[-1])
                
                num_generated_cols = flattened_data.shape[1]
                base_cols = st.session_state.target_features if st.session_state.target_features else []
                
                if len(base_cols) >= num_generated_cols:
                    columns = base_cols[:num_generated_cols]
                else:
                    extra_cols = [f"Generated_Feature_{i}" for i in range(len(base_cols), num_generated_cols)]
                    columns = base_cols + extra_cols
                
                df_syn = pd.DataFrame(flattened_data, columns=columns)
                st.session_state.synthetic_df = df_syn
                
                st.dataframe(df_syn, use_container_width=True)
                st.download_button("💾 Download CSV", data=df_syn.to_csv(index=False).encode('utf-8'), file_name='synthetic_zero_day.csv')
            else:
                st.warning(res.json().get("detail", "Error generating data."))
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")

# --- TAB 4: Agentic RAG Analyst ---
with tab_agent:
    st.subheader("Cybersecurity AI Agent (Free Local Ollama Integration)")
    
    model_choice = st.selectbox("Select Local Ollama Model:", ["llama3", "mistral", "phi3", "gemma"])
    
    if st.session_state.synthetic_df is None:
        st.warning("Generate synthetic data in Step 3 first.")
    elif not AGENTIC_AI_AVAILABLE:
        st.error(f"❌ Import Failed: {import_error_message}")
    else:
        user_query = st.text_input("Ask the AI about the synthetic zero-day flow:")
        if st.button("Query Agent") and user_query:
            with st.spinner("Analyzing locally via Ollama..."):
                try:
                    # Initialize local free LLM instance safely
                    llm = ChatOllama(model=model_choice, temperature=0)
                    
                    # Convert a relevant metadata preview of the dataframe to markdown text context
                    df_context = (
                        f"Dataframe Shape: {st.session_state.synthetic_df.shape}\n"
                        f"Columns Available: {list(st.session_state.synthetic_df.columns)}\n\n"
                        f"Dataframe Head Preview:\n{st.session_state.synthetic_df.head(5).to_markdown()}"
                    )
                    
                    system_prompt = (
                        "You are a Senior Cybersecurity Analyst. You are given a pandas dataframe named `df` containing "
                        "synthetic zero-day network traffic generated by an adversarial digital twin platform.\n\n"
                        f"{df_context}\n\n"
                        "Answer the user's question completely based on the dataset properties, patterns, or structure details provided above."
                    )
                    
                    # Execute a direct structured chat loop invocation (bypassing unstable ReAct parsing entirely)
                    response = llm.invoke(f"{system_prompt}\n\nUser Question: {user_query}")
                    
                    st.markdown("### Analyst Assessment")
                    st.write(response.content)
                    
                except Exception as e:
                    st.error(f"Execution Error: {e}")
