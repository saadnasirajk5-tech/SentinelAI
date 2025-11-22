# AI Question Answering Agent - Complete Code

## 🚀 Quick Start - Simple QA Agent (Recommended)

### Install Dependencies
```bash
pip install groq
```

### Basic QA Agent Code
```python
from groq import Groq

class SimpleQAAgent:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        
    def answer(self, question, context=""):
        """Answer a question with optional context"""
        if context:
            prompt = f"""Based on the context below, answer the question.

Context: {context}

Question: {question}

Answer:"""
        else:
            prompt = question
        
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content

# Usage
if __name__ == "__main__":
    # Get free API key at: https://console.groq.com
    agent = SimpleQAAgent(api_key="YOUR_GROQ_API_KEY")
    
    # Example 1: General question
    answer = agent.answer("What is machine learning?")
    print(f"Answer: {answer}\n")
    
    # Example 2: Question with context
    context = "Python is a programming language created by Guido van Rossum in 1991."
    answer = agent.answer("When was Python created?", context)
    print(f"Answer: {answer}")
```

---

## 🧠 RAG QA Agent (Document-Based)

### Install Dependencies
```bash
pip install groq sentence-transformers faiss-cpu numpy
```

### Complete RAG Implementation
```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

class RAGQAAgent:
    def __init__(self, groq_api_key):
        """Initialize RAG QA Agent with free tools"""
        # Free embedding model (runs locally, no API needed)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Free LLM via Groq API
        self.llm = Groq(api_key=groq_api_key)
        
        # FAISS vector store (free, super fast)
        self.index = None
        self.documents = []
        
    def add_documents(self, documents):
        """Add documents to knowledge base"""
        self.documents.extend(documents)
        
        # Create embeddings for all documents
        embeddings = self.embedder.encode(documents)
        
        # Initialize or update FAISS index
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings.astype('float32'))
        print(f"✅ Added {len(documents)} documents to knowledge base")
        
    def retrieve(self, query, k=3):
        """Retrieve top-k most relevant documents"""
        if self.index is None or len(self.documents) == 0:
            return []
            
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        return [self.documents[i] for i in indices[0]]
    
    def answer(self, question, return_sources=True):
        """Answer question using RAG"""
        # Step 1: Retrieve relevant context
        relevant_docs = self.retrieve(question, k=3)
        
        if not relevant_docs:
            return "No knowledge base found. Please add documents first."
        
        context = "\n\n".join(relevant_docs)
        
        # Step 2: Generate answer with LLM
        prompt = f"""Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        
        answer = response.choices[0].message.content
        
        if return_sources:
            return {
                'answer': answer,
                'sources': relevant_docs
            }
        return answer

# Usage Example
if __name__ == "__main__":
    # Initialize agent
    agent = RAGQAAgent(groq_api_key="YOUR_GROQ_API_KEY")
    
    # Add your knowledge base
    documents = [
        "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "Deep learning uses multiple layers of neural networks to learn hierarchical representations of data.",
        "Natural Language Processing (NLP) helps computers understand, interpret, and generate human language.",
        "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab.",
        "TensorFlow is an end-to-end machine learning platform developed by Google Brain team."
    ]
    
    agent.add_documents(documents)
    
    # Ask questions
    questions = [
        "When was Python created?",
        "What is machine learning?",
        "Who developed PyTorch?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = agent.answer(question)
        print(f"✅ Answer: {result['answer']}")
        print(f"📚 Sources: {len(result['sources'])} documents used")
```

---

## 📄 Load Documents from Files

```python
import PyPDF2
import os

def load_pdf(pdf_path):
    """Load text from PDF file"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_txt(txt_path):
    """Load text from TXT file"""
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_folder(folder_path):
    """Load all text/pdf files from a folder"""
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        if filename.endswith('.txt'):
            documents.append(load_txt(filepath))
        elif filename.endswith('.pdf'):
            documents.append(load_pdf(filepath))
    
    return documents

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

# Usage with RAG Agent
if __name__ == "__main__":
    agent = RAGQAAgent(groq_api_key="YOUR_GROQ_API_KEY")
    
    # Option 1: Load single PDF
    text = load_pdf("your_document.pdf")
    chunks = chunk_text(text)
    agent.add_documents(chunks)
    
    # Option 2: Load all files from folder
    # docs = load_folder("./documents")
    # all_chunks = []
    # for doc in docs:
    #     all_chunks.extend(chunk_text(doc))
    # agent.add_documents(all_chunks)
    
    # Now ask questions
    answer = agent.answer("Your question here")
    print(answer)
```

---

## 🔗 LangChain Version (Advanced)

### Install Dependencies
```bash
pip install langchain langchain-groq langchain-community faiss-cpu sentence-transformers
```

### LangChain QA Agent
```python
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

class LangChainQAAgent:
    def __init__(self, groq_api_key):
        """Initialize LangChain QA Agent"""
        # Free LLM via Groq
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )
        
        # Free embeddings (runs locally)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.vectorstore = None
        self.qa_chain = None
        
    def create_knowledge_base(self, documents):
        """Create vector store from documents"""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        texts = text_splitter.create_documents(documents)
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            texts,
            self.embeddings
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        
        print("✅ Knowledge base created with LangChain!")
        
    def answer(self, question):
        """Answer question with sources"""
        if self.qa_chain is None:
            return "Please create knowledge base first using create_knowledge_base()"
            
        result = self.qa_chain.invoke({"query": question})
        
        return {
            'answer': result['result'],
            'sources': [doc.page_content for doc in result['source_documents']]
        }

# Usage
if __name__ == "__main__":
    agent = LangChainQAAgent(groq_api_key="YOUR_GROQ_API_KEY")
    
    # Add documents
    documents = [
        "PyTorch is a deep learning framework developed by Facebook's AI Research lab.",
        "TensorFlow is Google's open-source machine learning framework.",
        "Transformers are the backbone architecture of modern NLP models like BERT and GPT.",
        "BERT stands for Bidirectional Encoder Representations from Transformers."
    ]
    
    agent.create_knowledge_base(documents)
    
    # Ask questions
    result = agent.answer("What is PyTorch?")
    print(f"Answer: {result['answer']}")
    print(f"\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source[:100]}...")
```

---

## 🎯 Choose Your Implementation

| Version | Best For | Complexity | Features |
|---------|----------|------------|----------|
| **Simple QA** | Quick answers, general questions | ⭐ Easy | Direct Q&A |
| **RAG QA** | Document-based answers | ⭐⭐ Medium | Retrieval + Generation |
| **LangChain** | Advanced features, tools | ⭐⭐⭐ Advanced | Chains, memory, tools |

---

## 🔑 Getting Free API Keys

### Groq (Recommended - Fastest)
1. Go to: https://console.groq.com
2. Sign up (free)
3. Create API key
4. Get 14,400 requests/day FREE

### Alternative: Ollama (100% Local)
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.3

# Then use in Python:
pip install ollama

import ollama
response = ollama.chat(model='llama3.3', messages=[
    {'role': 'user', 'content': 'What is AI?'}
])
print(response['message']['content'])
```

---

## 💡 Tips for Your Use Case

Based on your background (AI engineering student, building projects):

1. **Start with RAG QA Agent** - Most practical for real projects
2. **Works on Google Colab** - Free GPU for embeddings
3. **Portfolio project** - Deploy with Streamlit/FastAPI
4. **Freelancing** - Clients love document Q&A systems
5. **Low resource** - All models work on free tier

## 🚀 Next Steps

1. Get Groq API key (5 minutes)
2. Copy RAG QA Agent code
3. Test with your own documents
4. Add to your GitHub portfolio
5. Deploy for clients/projects

All code is production-ready and uses 100% free tools! 🎉
