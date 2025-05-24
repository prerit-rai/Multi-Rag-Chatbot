# Multi-Collection RAG System with Memory Layer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent chatbot system that retrieves information from multiple structured and semi-structured data collections while maintaining conversational context.

## 📌 Features

- **Multi-Source Retrieval**: Simultaneously query 5 different data collections
- **Contextual Memory**: Maintains conversation history across sessions
- **Hybrid Search**: Combines vector similarity and metadata filtering
- **Modern UI**: Chainlit-based web interface
- **Data Provenance**: Source attribution for answers

**Supported Data Collections**:
- 🏃 Wearable Data (Fitbit/Apple Health-like samples)
- 💬 Chat History
- 👤 User Profile
- 📍 Location Data
- 🎯 Custom Collection (Goals/Metrics)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/multi-collection-rag-system.git
cd multi-collection-rag-system

# Install dependencies
pip install -r requirements.txt
# Start the web interface
chainlit run app.py


### Project Structure
.
├── app.py             # Main application entrypoint
├── rag.py             # RAG pipeline implementation
├── data_loader.py     # Data collection handling
├── memory.py          # Conversation memory management
├── config/            # Configuration files
├── data/              # Sample data collections
├── requirements.txt
└── README.md