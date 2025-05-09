# Detect file type, if is pdf, then convert to txt, if its txt, then start chunking and keyword tagging, and save to db
import os
import sqlite3
import fitz
import jieba
import nltk
import re
from tqdm import tqdm
from config import DB_PATH
from langdetect import detect
from collections import Counter
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from langdetect.lang_detect_exception import LangDetectException
from langchain.text_splitter import RecursiveCharacterTextSplitter

def detect_file_type(file):
    file_type = os.path.splitext(file)
    return file_type[1]

def split_text_into_chunks(text, chunk_size = 700, chunk_overlap = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", ".", "!", "！", "，", "?", "？","。"]
    )
    return text_splitter.split_text(text)

def generate_chinese_tags(text, top_k = 5):
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z ]", "", text)
    words = jieba.lcut(text) + nltk.word_tokenize(text)
    counter = Counter(w for w in words if len(w) > 1)
    common = counter.most_common(top_k)
    return [word for word, _ in common]

def create_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path) # Do not pass DB=path into this
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_index INTEGER,
            chunks TEXT,
            tags TEXT
                   )
    ''')
    conn.commit
    return conn

def extract_text(file):
    if detect_file_type(file) == ".pdf":
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n\n"
    elif detect_file_type(file) == ".txt":
        with open(file, "r", encoding = "utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Please only use .pdf or .txt file!")
    return text

# Main pipeline: read file --> chunk file --> extract keywords --> save to database
model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model)

def file_to_db(file, db_path = DB_PATH):
    text = extract_text(file)
    print(f"Processing file: {file}")

    chunks = split_text_into_chunks(text)
    print(f"Text split into {len(chunks)} chunks")

    tags_list = []
    for chunk in tqdm(chunks, desc = "Extracting tags"):
        chunk = chunk.strip()
        
        if not chunk:
            tags_list.append([])  # Optional: store empty tags for empty chunk
            continue

        try:
            lang = detect(chunk)
        except LangDetectException:
            lang = "unknown"
        except Exception as e:
            print(f"Language detection error: {e}")
            lang = "unknown"

        if lang == "zh-cn":
            tags = generate_chinese_tags(chunk)
        else:
            keywords = kw_model.extract_keywords(chunk, top_n=5)
            tags = [kw for kw, _ in keywords]

        tags_list.append(tags)

    conn = create_db(db_path)
    cursor = conn.cursor()
    insert_data = []

    for i, (chunk, tags) in tqdm(enumerate(zip(chunks, tags_list)), total = len(chunks), desc = "Saving to DB"):
        insert_data.append((i,chunk,",".join(tags)))

    cursor.executemany('''
        INSERT INTO document (chunk_index, chunks, tags) VALUES (?,?,?)''', insert_data)
        
    conn.commit()
    conn.close()
    print(f"Mission Complete! Data has been inserted with {len(insert_data)} chunks")

#-------------

# Read through data base, embedding, then build faiss index file for similarity research 
# will be using OpenAi API for embedding, use LangChain to integreate OpenAi API

import sqlite3
import os
from tqdm import tqdm
from config import DB_PATH, OPENAI_API_KEY
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # This is an environment variable, it allows LangChain and OpenAI's python SDK to look for the API key in this environment

embedding_model = OpenAIEmbeddings() # Creates an embedding model wrapper in LangChain that uses OpenAI's text-embedding models

def load_all_from_db(db_path = DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_index, chunks, tags FROM document")
    rows = cursor.fetchall()
    conn.close()

    # The structure below will help my search.py extract chunks, and tags from chunks metadata
    document = []
    for chunk_index, chunks, tags in tqdm(rows, desc= "Building document list"):
        metadata = {
            "chunk_index": chunk_index,
            "tags": tags
        }
        
        document.append(Document(page_content = chunks, metadata=metadata)) # Reads all stored chunked text + tags, and wraps each into a LangChain Document with metadata
    return document

def build_faiss_index(document, save_path = "vector_index"):
    print("Vectorizing contents...")
    vector_store = FAISS.from_documents(document, embedding_model) # Here I used OpenAI model

    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"Vector index has been saved to {save_path}")

#------
# Load local FAISS index storage, conduct similarity search based on user's query, then create an efficient GPT prompt for OpenAI Chat API
# Before searching similar content in FAISS vector index storage, I need to first embed user's inputr, which allows the program to search similair stuff in FAISS index storage

import os
import langid
from config import DB_PATH, OPENAI_API_KEY
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
embedding_model = OpenAIEmbeddings()

def load_vector_storage(path="vector_index"):
    return FAISS.load_local(path,embedding_model, allow_dangerous_deserialization=True)

# k=5 means retrieve the top 5 similair chunks
def search_similar_chunks(query, k=5):
    vector_store = load_vector_storage()
    return vector_store.similarity_search(query, k=k)

# Originally, I used LangChain's detect method to identify the user's language, but it misrecognized Chinese as Korean,
# so I switched to langid, which is a more accurate language identifier.

def detect_language(text):
    try:
        lang, _ = langid.classify(text)
        return lang  # e.g., "zh", "en", "ko", etc.
    except:
        return "unknown"

def answer_question_with_prompt(query, retrieved_chunks):
    if not retrieved_chunks:
        return "No releated information can be found! Please try another question."
    
    # Below are three information pieces that will be passed to OpenAI's gpt later

    # This is a list of retrieved chunks based on similarity search
    context_chunks = "\n\n".join(doc.page_content for doc in retrieved_chunks)

    # This is a list of tags of chunks above, extracted from the metadata of those chunks
    # By structuring it using f-string, it will feed these tags to gpt as bullet points
    tags = [f"- {doc.metadata.get('tags', '')}" for doc in retrieved_chunks]

    # Below join all tags above into one string but still maintain their bullet point format
    joined_tags = "\n".join(tags)


    # Prompt starts below:
    language = detect_language(query)
    prompt = f"""You are an intelligent AI assistant built for my RAG Chatbot system. Your duty is to answer the user's question based on the content and tags below, and you need to answer the question with this language the user is using: {language}.
    
    Guidelines for you:
    1. Base your answer ONLY on the provided content and tags, however, if the answer is not explicityly stated, you are allowed to summarize, infer, or connect clues across different parts.
    2. If relevant information exists in any form, do your best to provide a helpful answer that might answer user's question
    3. Only if the content is indeed completely irrelevant with user's question, respond with: "There is no relevant information in the provided content" in user's language: {language}.
    4. If user ask anything about you, for example:"Who are you" or "What is your name", you should answer that:"I am your AI assistant!"

    Tags:
    {joined_tags}

    Content Chunks:
    {context_chunks}

    User's Question: {query}

    Answer:"""

    # Below I set the temperature and model of gpt
    # Temperature means: [Temperature controls randomness or creativity of the model's output.]
    # For instance: temperature = 0.0 → Deterministic, logical, focused. Always gives the most likely next word. temperature >= 1.0 → More creative, varied, and potentially unexpected responses.
    chat = ChatOpenAI(temperature = 0.5, model = "gpt-4-1106-preview")
    response = chat.invoke(prompt)
    return response.content.strip()
