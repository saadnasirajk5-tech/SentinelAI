
"""
AI Question Answering Agent - GitHub Safe Version
No hardcoded API keys - Safe to push to GitHub!
"""

# ============================================================================
# INSTALLATION (Run in Colab first cell)
# ============================================================================
# !pip install -q groq sentence-transformers faiss-cpu PyPDF2

# ============================================================================
# IMPORTS
# ============================================================================
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import os


# ============================================================================
# API KEY SETUP (Safe method - uses environment variable)
# ============================================================================
# Set your API key in Colab with this command BEFORE running the code:
# import os
# os.environ["GROQ_API_KEY"] = "your-actual-api-key-here"

API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================================
# 1. SIMPLE QA AGENT
# ============================================================================
class SimpleQAAgent:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        
    def answer(self, question, context=""):
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


# ============================================================================
# 2. RAG QA AGENT (Document-Based)
# ============================================================================
class RAGQAAgent:
    def __init__(self, groq_api_key):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = Groq(api_key=groq_api_key)
        self.index = None
        self.documents = []
        
    def add_documents(self, documents):
        self.documents.extend(documents)
        embeddings = self.embedder.encode(documents)
        
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings.astype('float32'))
        print(f"✅ Added {len(documents)} documents to knowledge base")
        
    def retrieve(self, query, k=3):
        if self.index is None or len(self.documents) == 0:
            return []
            
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        return [self.documents[i] for i in indices[0]]
    
    def answer(self, question, return_sources=True):
        relevant_docs = self.retrieve(question, k=3)
        
        if not relevant_docs:
            return "No knowledge base found. Please add documents first."
        
        context = "\n\n".join(relevant_docs)
        
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


# ============================================================================
# 3. DOCUMENT LOADERS
# ============================================================================
def load_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def load_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()


def load_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        if filename.endswith('.txt'):
            documents.append(load_txt(filepath))
        elif filename.endswith('.pdf'):
            documents.append(load_pdf(filepath))
    
    return documents


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


# ============================================================================
# 4. USAGE EXAMPLES
# ============================================================================
def example_simple_qa(api_key):
    print("\n" + "="*70)
    print("SIMPLE QA AGENT EXAMPLE")
    print("="*70 + "\n")
    
    agent = SimpleQAAgent(api_key=api_key)
    
    # General question
    print("Question: What is machine learning?")
    answer = agent.answer("What is machine learning?")
    print(f"Answer: {answer}\n")
    
    # Question with context
    context = "Python is a programming language created by Guido van Rossum in 1991."
    print(f"Context: {context}")
    print("Question: When was Python created?")
    answer = agent.answer("When was Python created?", context)
    print(f"Answer: {answer}\n")


def example_rag_qa(api_key):
    print("\n" + "="*70)
    print("RAG QA AGENT EXAMPLE")
    print("="*70 + "\n")
    
    agent = RAGQAAgent(groq_api_key=api_key)
    
    # Add knowledge base
    documents = [
        "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "Deep learning uses multiple layers of neural networks to learn hierarchical representations of data.",
        "Natural Language Processing (NLP) helps computers understand, interpret, and generate human language.",
        "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab.",
        "TensorFlow is an end-to-end machine learning platform developed by Google Brain team.",
        "Transformers are the backbone architecture of modern NLP models like BERT and GPT.",
        "BERT stands for Bidirectional Encoder Representations from Transformers, developed by Google.",
        "GPT (Generative Pre-trained Transformer) is a language model architecture developed by OpenAI."
    ]
    
    agent.add_documents(documents)
    
    # Ask questions
    questions = [
        "When was Python created?",
        "What is machine learning?",
        "Who developed PyTorch?",
        "What does BERT stand for?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = agent.answer(question)
        print(f"✅ Answer: {result['answer']}")
        print(f"📚 Sources used: {len(result['sources'])} documents")


def example_pdf_loading(api_key):
    print("\n" + "="*70)
    print("PDF LOADING EXAMPLE")
    print("="*70 + "\n")
    
    agent = RAGQAAgent(groq_api_key=api_key)
    
    # Load single PDF
    try:
        text = load_pdf("your_document.pdf")
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        agent.add_documents(chunks)
        
        # Ask questions about the PDF
        result = agent.answer("What is the main topic of this document?")
        print(f"Answer: {result['answer']}")
    except FileNotFoundError:
        print("⚠️  PDF file not found. Update path to your PDF.")
    except Exception as e:
        print(f"⚠️  Error: {e}")


def example_folder_loading(api_key):
    print("\n" + "="*70)
    print("FOLDER LOADING EXAMPLE")
    print("="*70 + "\n")
    
    agent = RAGQAAgent(groq_api_key=api_key)
    
    # Load all documents from a folder
    try:
        docs = load_folder("./documents")
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_text(doc, chunk_size=500, overlap=50))
        
        agent.add_documents(all_chunks)
        
        # Ask questions
        result = agent.answer("Summarize the key information from these documents")
        print(f"Answer: {result['answer']}")
    except FileNotFoundError:
        print("⚠️  Folder not found. Create './documents' folder with PDF/TXT files.")
    except Exception as e:
        print(f"⚠️  Error: {e}")


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n🤖 AI Question Answering Agent")
    print("="*70)
    
    # Check if API key is set
    if not API_KEY:
        print("\n⚠️  API KEY NOT FOUND!")
        print("\nTo use this code in Colab:")
        print("1. Get free API key: https://console.groq.com")
        print("2. Add this at the TOP of your notebook:")
        print("   import os")
        print('   os.environ["GROQ_API_KEY"] = "your-actual-key-here"')
        print("\n3. Then run this code again!")
    else:
        print("✅ API key found! Running examples...\n")
        
        # Uncomment the example you want to run:
        example_simple_qa(API_KEY)
        example_rag_qa(API_KEY)
        # example_pdf_loading(API_KEY)
        # example_folder_loading(API_KEY)
        
        print("\n" + "="*70)
        print("✅ EXAMPLES COMPLETED!")
        print("="*70)