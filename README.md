# PDF-Based Conversational Chatbot with RAG

## Overview

This project implements a conversational chatbot that leverages Retrieval Augmented Generation (RAG) to provide accurate responses from PDF documents. The system is designed to help users access information from knowledge base documents through natural language queries, with a focus on maintaining context and providing source attribution.

## Key Features

1. **Multi-Document Support**
   - Handles multiple PDF documents simultaneously
   - Efficiently retrieves information from appropriate documents
   - Maintains document source tracking for transparency

2. **Contextual Response Generation**
   - Uses OpenAI's GPT-3.5 Turbo model for coherent responses
   - Strictly adheres to information present in uploaded documents
   - Implements conversation memory for contextual understanding

3. **Source Attribution**
   - Displays source document names for each response
   - Ensures transparency and traceability of information

4. **Conversation Management**
   - Maintains conversation history
   - Uses ConversationBufferMemory for context retention
   - Provides contextually relevant responses

5. **Advanced Document Processing**
   - Implements chunk-based document processing
   - Uses FAISS for efficient vector storage and retrieval
   - Employs HuggingFace's all-MiniLM-L6-v2 for embeddings

## Technical Architecture

### Components
1. **Document Processing**
   - PDF text extraction using PyPDF2
   - Text chunking with CharacterTextSplitter
   - Document metadata management

2. **Vector Storage**
   - FAISS vector store implementation
   - Efficient similarity search capabilities
   - Source document tracking

3. **Language Model Integration**
   - OpenAI GPT-3.5 Turbo integration
   - Custom prompt templates
   - Conversation chain management

4. **User Interface**
   - Streamlit-based web interface
   - File upload capabilities
   - Interactive chat interface

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/ChatPDF.git
   cd ChatPDF
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**
   Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage Guide

1. **Document Upload**
   - Upload one or more PDF files using the sidebar
   - Click "Process" to initialize the system

2. **Querying**
   - Type questions in the chat input
   - View responses with source attribution
   - Continue conversation with context-aware follow-up questions

## Implementation Details

- **Document Processing**: Files are processed into chunks of 1000 tokens with 200 token overlap
- **Embedding Model**: Utilizes all-MiniLM-L6-v2 from HuggingFace
- **Vector Store**: Implements FAISS for efficient similarity search
- **LLM Integration**: Uses OpenAI's GPT-3.5 Turbo with temperature 0.2
- **Memory Management**: Implements ConversationBufferMemory for context retention

## Limitations and Future Improvements

1. **Current Limitations**
   - Limited to PDF file format
   - Requires manual document processing
   - Question suggestions feature needs improvement

2. **Planned Improvements**
   - Support for additional document formats
   - Automated document processing
   - Enhanced question suggestion system
   - Improved source attribution

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue in the repository.
