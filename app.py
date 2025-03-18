# app.py - Main Flask application
from flask import Flask, render_template, request, jsonify, session, send_file, g
import os
import io
import json
import base64
import requests
import traceback
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import uuid
import pickle
import tempfile

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PDFOllamaProcessor:
    def __init__(self):
        """Initialize the PDF processor"""
        self.model = None
    
    def extract_text_with_ocr(self, pdf_file):
        """Extract text from PDF using both text extraction and OCR"""
        # Read the PDF file content
        pdf_content = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer
        
        # First, try standard text extraction
        text = self.extract_text_from_pdf(pdf_file)
        
        # If no text extracted, use OCR
        if not text.strip():
            text = self.ocr_pdf(io.BytesIO(pdf_content))
        
        return text
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text using PyPDF2 text extraction"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text
    
    def ocr_pdf(self, pdf_content):
        """Perform OCR on PDF pages"""
        images = convert_from_bytes(pdf_content.read())
        
        full_text = ''
        for image in images:
            page_text = pytesseract.image_to_string(image)
            full_text += page_text + '\n'
        
        return full_text
    
    def split_text_into_chunks(self, text, chunk_size=200, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, chunks):
        """Create embeddings from text chunks"""
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = self.model.encode(chunks)
        return embeddings
    
    def find_similar_entries(self, query, database_df, top_n=3):
        """Find the most similar entries in the vector database"""
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
        # Embed the query
        query_embedding = self.model.encode(query)
    
        similarities = []
        for index, row in database_df.iterrows():
            # Convert embeddings to numpy arrays
            embedding = np.array(row["embedding"], dtype=np.float32)
            query_vec = np.array(query_embedding, dtype=np.float32)
        
            # Ensure both are 1D arrays of the same length
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            if query_vec.ndim > 1:
                query_vec = query_vec.flatten()
        
            similarity = self.cosine_similarity(query_vec, embedding)
            similarities.append((index, similarity))
    
        # Sort by similarity in descending order
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        top_indices = [index for index, _ in sorted_similarities[:top_n]]
        return database_df.loc[top_indices]
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate the cosine similarity between two vectors"""
        # Ensure both vectors are 1D numpy arrays
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()
    
        # Ensure vectors are the same length by truncating to the shorter length
        min_length = min(len(vec1), len(vec2))
        vec1 = vec1[:min_length]
        vec2 = vec2[:min_length]
    
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
    
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
    
        return dot_product / (norm_vec1 * norm_vec2)
    
    def query_ollama(self, model_name, prompt, system_instructions="", ollama_url="http://localhost:11434"):
        """Queries the Ollama model using the HTTP API"""
        try:
            # Prepare the API endpoint URL
            url = f"{ollama_url}/api/generate"
            
            # Prepare the request payload
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False  # Get complete response at once
            }
            
            # Add system instructions if provided
            if system_instructions:
                data["system"] = system_instructions
            
            # Send POST request to Ollama API
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response received')
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Please make sure Ollama is running."
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize processor
processor = PDFOllamaProcessor()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    """Process uploaded PDF and create vector database"""
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    pdf_file = request.files['pdf_file']
    
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if pdf_file:
        try:
            # Extract text from PDF
            text = processor.extract_text_with_ocr(pdf_file)
            
            # Split text into chunks
            chunks = processor.split_text_into_chunks(text)
            
            # Create embeddings
            embeddings = processor.create_embeddings(chunks)
            
            # Create database
            database = []
            for i in range(len(chunks)):
                database.append({
                    'text': chunks[i],
                    'embedding': embeddings[i].tolist()
                })
            
            # Generate a unique ID for this database
            db_id = str(uuid.uuid4())
            
            # Store database in a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            with open(temp_file.name, 'wb') as f:
                pickle.dump(database, f)
            
            # Store the temporary file path in the session
            session['temp_db_path'] = temp_file.name
            
            # Convert to DataFrame for CSV export
            df = pd.DataFrame(database)
            
            # Generate a downloadable CSV file
            csv_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            df.to_csv(csv_file.name, index=False)
            
            # Store the CSV file path in the session
            session['csv_db_path'] = csv_file.name
            
            return jsonify({
                'success': True,
                'message': 'Vector database created successfully',
                'chunk_count': len(chunks),
                'db_id': db_id
            })
        
        except Exception as e:
            return jsonify({
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500

@app.route('/download-database', methods=['GET'])
def download_database():
    """Download the vector database as a CSV file"""
    try:
        csv_path = session.get('csv_db_path')

        if not csv_path or not os.path.exists(csv_path):
            return jsonify({'error': 'Database file not found'}), 404

        return send_file(
            csv_path,
            as_attachment=True,
            download_name='vector_database.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/export-database', methods=['GET'])
def export_database():
    """Export the vector database as CSV"""
    if 'temp_db_path' not in session:
        return jsonify({'error': 'No database to export'}), 400
    
    try:
        # Load database from temporary file
        with open(session['temp_db_path'], 'rb') as f:
            database = pickle.load(f)
        
        # Create DataFrame
        df = pd.DataFrame(database)
        
        # Convert DataFrame to CSV
        csv_data = df.to_csv(index=False)
        
        # Return CSV as attachment
        response = {
            'success': True,
            'filename': 'vector_database.csv',
            'csv_data': base64.b64encode(csv_data.encode()).decode('utf-8')
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load-database', methods=['POST'])
def load_database():
    """Load vector database from CSV"""
    if 'csv_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    csv_file = request.files['csv_file']
    
    if csv_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if csv_file:
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            if 'text' not in df.columns or 'embedding' not in df.columns:
                return jsonify({'error': 'CSV must contain text and embedding columns'}), 400
            
            # Convert JSON strings to lists if needed
            if isinstance(df.iloc[0]['embedding'], str):
                df['embedding'] = df['embedding'].apply(json.loads)
            
            # Convert to list of dicts
            database = df.to_dict('records')
            
            # Store database in a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            with open(temp_file.name, 'wb') as f:
                pickle.dump(database, f)
            
            # Store the temporary file path in the session
            session['temp_db_path'] = temp_file.name
            
            return jsonify({
                'success': True,
                'message': 'Database loaded successfully',
                'chunk_count': len(df)
            })
        
        except Exception as e:
            return jsonify({
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500

@app.route('/query', methods=['POST'])
def query():
    """Query the vector database and Ollama"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    query_text = data.get('query')
    model_name = data.get('model')
    system_instructions = data.get('system', "")
    ollama_url = data.get('ollama_url', 'http://localhost:11434')
    
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400
    
    if not model_name:
        return jsonify({'error': 'No model name provided'}), 400
    
    if 'temp_db_path' not in session:
        return jsonify({'error': 'No database loaded'}), 400
    
    try:
        # Load database from temporary file
        with open(session['temp_db_path'], 'rb') as f:
            database = pickle.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(database)
        
        # Find similar entries
        similar_df = processor.find_similar_entries(query_text, df)
        
        # Combine texts and create prompt
        context = " ".join(similar_df["text"].values) if not similar_df.empty else ""
        prompt = f"""Context: {context}

Query: {query_text}

Please provide a relevant response based on the context and query above."""
        
        # Query Ollama
        ollama_response = processor.query_ollama(model_name, prompt, system_instructions, ollama_url)
        
        # Return results
        return jsonify({
            'success': True,
            'context': similar_df.to_dict('records'),
            'ollama_response': ollama_response
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Clean up temporary files when the application stops
temp_files = {}

@app.teardown_appcontext
def cleanup_temp_files(exc=None):
    session_id = getattr(g, 'session_id', None)  # Retrieve session ID safely
    if session_id and session_id in temp_files:
        del temp_files[session_id]  # Clean up files associated with session ID
                
@app.after_request
def cleanup(response):
    cleanup_temp_files()
    return response

@app.before_request
def store_session_id():
    g.session_id = session.get('_id', None)  # Ensure it handles cases where '_id' is not set



if __name__ == '__main__':
    app.run(debug=True)