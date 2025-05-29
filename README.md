# PeoplePlus - Advanced RAG Chatbot

An intelligent document-based chatbot powered by Retrieval-Augmented Generation (RAG) with hybrid search capabilities. This system combines semantic and keyword-based search to provide accurate answers from your document collection.

## 🚀 Features

- **Hybrid Search**: Combines dense (semantic) and sparse (BM25) search for optimal retrieval
- **Query Enhancement**: Automatically expands and decomposes user queries for better results
- **Chat History**: Maintains conversation context with persistent storage
- **User Management**: Secure user authentication and session management
- **Document Processing**: Automated PDF document ingestion and chunking
- **Web Interface**: Clean, responsive chat interface built with FastAPI
- **Vector Database**: Powered by Pinecone for scalable vector search
- **Advanced Ranking**: Document re-ranking based on relevance scores

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI
- **LLM**: Groq (Llama 3.1 8B Instant)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Database**: Pinecone
- **Traditional Database**: PostgreSQL
- **Document Processing**: LangChain, PyPDF
- **Search**: BM25 + Semantic Hybrid Search
- **Frontend**: Jinja2 Templates, HTML/CSS/JavaScript
- **ORM**: SQLAlchemy

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL database
- API keys for:
  - Pinecone
  - Groq
  - HuggingFace (optional, for embeddings)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PulkitBansal2/PeoplePlus.git
   cd PeoplePlus
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file with the following variables:
   ```env
   # Database Configuration
   USER=your_postgres_username
   PASSWORD=your_postgres_password
   HOST=localhost
   PORT=5432
   DB_NAME=peopleplus_db
   
   # API Keys
   PINECONE_API_KEY=your_pinecone_api_key
   GROQ_API_KEY=your_groq_api_key
   HF_TOKEN=your_huggingface_token
   ```

4. **Document Preparation**
   ```bash
   # Create data directory and add your PDF documents
   mkdir data
   # Place your PDF files in the data/ directory
   ```

5. **Database Setup**
   ```bash
   # The application will automatically create the database and tables
   python main.py  # This will initialize the vector store and database
   ```

6. **Run the application**
   ```bash
   # Start the web server
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

## 🖥️ Usage

### Web Interface

1. **Access the Application**
   - Navigate to `http://localhost:8000`
   - You'll be redirected to the login page

2. **User Registration/Login**
   - Create a new account or login with existing credentials
   - The system supports automatic user registration

3. **Chat Interface**
   - Ask questions about your documents
   - View chat history in the sidebar
   - The system maintains conversation context

### Command Line Interface

```bash
# Run the chatbot in terminal mode
python main.py
```

## 🏗️ Architecture

### RAG Pipeline

1. **Query Processing**
   - Query expansion with synonyms and context
   - Query decomposition into sub-queries
   - History-aware query reformulation

2. **Document Retrieval**
   - Hybrid search (BM25 + semantic similarity)
   - Document deduplication and re-ranking
   - Context formatting for LLM

3. **Answer Generation**
   - Context-aware response generation
   - Chat history integration
   - Response storage and tracking

### Data Models

- **User**: Authentication and session management
- **ChatHistory**: Conversation persistence and context

## 📁 Project Structure

```
PeoplePlus/
├── main.py              # Core RAG pipeline and CLI interface
├── app.py               # FastAPI web application
├── templates/           # HTML templates for web interface
├── data/               # PDF documents directory
├── bm25_values.json    # BM25 encoder cache
├── requirements.txt    # Python dependencies
└── .env               # Environment variables
```

## 🔐 Security Features

- Session-based authentication
- Password hashing (Note: Consider using bcrypt for production)
- Environment variable protection for API keys
- User isolation for chat histories

## 🧪 Testing

```bash
# Test the RAG pipeline
python -c "from main import rag_pipeline; print(rag_pipeline('your test query'))"

# Test web interface
curl -X POST "http://localhost:8000/chat" -d "query=test question"
```

## 📈 Performance Optimization

- **Chunking Strategy**: 500 characters with 250 overlap
- **Embedding Caching**: Persistent vector storage in Pinecone
- **BM25 Caching**: Serialized sparse encoder
- **Query Optimization**: Intelligent query expansion and decomposition

## 🚀 Deployment

### Local Development
```bash
uvicorn app:app --reload
```

### Production
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Recommended)
```dockerfile
# Create Dockerfile for containerized deployment
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## 📄 API Endpoints

- `GET /` - Redirect to login
- `GET /login` - Login form
- `POST /login` - Login submission
- `GET /create_user` - User registration form
- `POST /create_user` - User registration
- `GET /chat` - Chat interface
- `POST /chat` - Submit chat query
- `GET /logout` - User logout

## 🐛 Known Issues

- Password hashing uses simple string comparison (upgrade to bcrypt recommended)
- No file upload interface (documents must be manually added to data/ folder)
- Limited error handling for API rate limits

## 📈 Roadmap

- [ ] Advanced password hashing (bcrypt/argon2)
- [ ] File upload interface for documents
- [ ] Real-time chat with WebSocket support
- [ ] Multi-document collection management
- [ ] Advanced analytics dashboard
- [ ] Mobile-responsive design improvements
- [ ] API rate limiting and caching
- [ ] Docker containerization
- [ ] Kubernetes deployment configs

## 🙏 Acknowledgments

- LangChain community for RAG framework
- Pinecone for vector database services
- Groq for fast LLM inference
- HuggingFace for embedding models
- FastAPI for the excellent web framework

---

⭐ **Star this repository if you find it helpful!**

## 🔧 Quick Setup Commands

```bash
# Complete setup in one go
git clone https://github.com/PulkitBansal2/PeoplePlus.git
cd PeoplePlus
pip install -r requirements.txt
# Add your .env file and PDF documents
python main.py  # Initialize the system
uvicorn app:app --reload  # Start the web server
```
