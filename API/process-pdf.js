// This file would be placed in /netlify/functions/process-pdf.js if using Netlify
// Or in /api/process-pdf.js if using Vercel

const { SentenceTransformer } = require('sentence-transformers');
const pdf = require('pdf-parse');
const axios = require('axios');

// Initialize transformer model
let model;

async function getModel() {
  if (!model) {
    model = new SentenceTransformer('all-MiniLM-L6-v2');
  }
  return model;
}

// Helper function to split text into chunks
function splitTextIntoChunks(text, chunkSize = 200, overlap = 50) {
  const words = text.split(/\s+/);
  const chunks = [];
  
  for (let i = 0; i < words.length; i += (chunkSize - overlap)) {
    const chunk = words.slice(i, i + chunkSize).join(' ');
    chunks.push(chunk);
  }
  
  return chunks;
}

// Function to extract text from PDF
async function extractTextFromPdf(buffer) {
  try {
    const data = await pdf(buffer);
    return data.text;
  } catch (error) {
    console.error('Error extracting text from PDF:', error);
    throw new Error('Failed to extract text from PDF');
  }
}

// Main handler function
exports.handler = async function(event, context) {
  // Check if the request is a POST request
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }
  
  try {
    // Parse the multipart form data
    const formData = JSON.parse(event.body);
    
    // Get PDF file from form data (this depends on the serverless platform)
    // In a real implementation, this would access the uploaded file
    // For demonstration, assume we have the PDF buffer
    const pdfBuffer = formData.pdfBuffer; 
    
    // Extract text from PDF
    const text = await extractTextFromPdf(pdfBuffer);
    
    // Split text into chunks
    const chunks = splitTextIntoChunks(text);
    
    // Get model
    const model = await getModel();
    
    // Create embeddings
    const embeddings = await model.encode(chunks);
    
    // Create database
    const database = chunks.map((chunk, i) => ({
      text: chunk,
      embedding: Array.from(embeddings[i])
    }));
    
    return {
      statusCode: 200,
      body: JSON.stringify({
        success: true,
        message: 'Vector database created successfully',
        database: database,
        chunk_count: chunks.length
      })
    };
  } catch (error) {
    console.error('Error processing PDF:', error);
    
    return {
      statusCode: 500,
      body: JSON.stringify({
        error: error.message,
        stack: error.stack
      })
    };
  }
};

// This would be a separate function file: /api/query.js
exports.queryHandler = async function(event, context) {
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }
  
  try {
    const data = JSON.parse(event.body);
    const { query, model: modelName, system, ollama_url, database } = data;
    
    if (!query) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: 'No query provided' })
      };
    }
    
    if (!modelName) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: 'No model name provided' })
      };
    }
    
    if (!database || !Array.isArray(database)) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: 'Invalid or missing database' })
      };
    }
    
    // Get model for query embedding
    const sentenceModel = await getModel();
    
    // Embed the query
    const queryEmbedding = await sentenceModel.encode(query);
    
    // Find similar entries
    const similarities = [];
    database.forEach((entry, index) => {
      const embedding = new Float32Array(entry.embedding);
      
      // Calculate cosine similarity
      const dotProduct = embedding.reduce((sum, val, i) => sum + val * queryEmbedding[i], 0);
      const magnitude1 = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
      const magnitude2 = Math.sqrt(queryEmbedding.reduce((sum, val) => sum + val * val, 0));
      
      const similarity = magnitude1 > 0 && magnitude2 > 0 ? dotProduct / (magnitude1 * magnitude2) : 0;
      similarities.push({ index, similarity });
    });
    
    // Sort by similarity and get top 3
    similarities.sort((a, b) => b.similarity - a.similarity);
    const topIndices = similarities.slice(0, 3).map(item =>
