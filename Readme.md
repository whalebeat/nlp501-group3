# Create environment
python -m venv nlp_env

# Activate
nlp_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Ollama model
ollama pull qwen2.5:1.5b

# Build vector database
python build_index.py

# Run chatbot
streamlit run app.py