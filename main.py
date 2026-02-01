import google.generativeai as genai
import numpy as np
import tiktoken
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os
from fastapi import FastAPI, UploadFile, File
import io
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
################  TEXT extraction  ###############################
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index = None
chunks = None

def read_pdf_bytes(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def read_txt_bytes(file_bytes):
    return file_bytes.decode("utf-8")

def load_document_from_upload(filename, file_bytes):
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return read_pdf_bytes(file_bytes)
    elif ext == ".txt":
        return read_txt_bytes(file_bytes)
    else:
        raise ValueError("Unsupported file format")




##################   FAISS   ####################################
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=200, overlap=50):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks=[]
    step = chunk_size - overlap
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
#     # response = client.embeddings.create(
#     #      model="text-embedding-3-small",
#     #     input=chunks
#     # )

#     # embeddings = np.array(
#     #     [item.embedding for item in response.data],
#     #     dtype = "float32"
#     # )
    embeddings = model.encode(chunks)
    return embeddings.astype("float32")

# def embed_chunks(chunks):
#     embeddings = []

#     for chunk in chunks:
#         result = genai.embed_content(
#             model="gemini-embedding-001",
#             content=chunk
#         )
#         embeddings.append(result["embedding"])

#     return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def embed_query(query):
#     # response = client.embeddings.create(
#     #     model="text-embedding-3-small",
#     #     input = query
#     # )

#     # return np.array(
#     #     response.data[0].embedding,
#     #     dtype="float32"
#     # ).reshape(1,-1)
    embedding = model.encode([query])
    return embedding.astype("float32")

# def embed_query(query):
#     result = genai.embed_content(
#         model="gemini-embedding-001",
#         content=query
#     )
#     return np.array(result["embedding"], dtype="float32").reshape(1, -1)

def search(index, query_vector, chunks, top_k=5):
    distance, ind = index.search(query_vector, top_k)

    results = []
    for indx, dist in zip(ind[0], distance[0]):
        results.append({
            "text" : chunks[indx],
            "distance" : float(dist)
        })
    return results



@app.post("/upload")
async def post_meth(file: UploadFile = File(...)):
    file_bytes = await file.read()
    document = load_document_from_upload(file.filename, file_bytes)

    global index, chunks
    chunks = chunk_text(document)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    return {"status":"RAG started", "chunks_len" : len(chunks)}

# chunks = chunk_text(document)
# print(chunks)
# embeddings = embed_chunks(chunks)
# print(embeddings)
#index = build_faiss_index(embeddings)






############################   RAG    ####################################
genai.configure(api_key="GEMINI_API_KEY")

def build_context(results):
    context = ""
    for i, res in enumerate(results, 1):
        context += f"[{i}] {res['text']}\n"
    return context

def build_rag_prompt(context, question):
    system_prompt = (
        "You are an assistant that answers questions ONLY using the provided context. "
        "If the answer is not present in the context, say 'I don't know'."
    )

    user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""
    return system_prompt, user_prompt


def generate_answer(system_prompt, user_prompt):
    # Create the ACTUAL model object
    model = genai.GenerativeModel("models/gemini-2.5-flash") # gemini-1.5-flash is stable and fast

    prompt = system_prompt + "\n\n" + user_prompt

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0
        }
    )

    return response.text

# @app.post("/query")
# def query_rag(question:str):
#     if index is None:
#         return "No doc uploaded."
#     query_vector = embed_query(question)
#     results = search(index, query_vector, chunks, top_k=3)
#     context = build_context(results)
#     system_prompt, user_prompt = build_rag_prompt(context, question)
#     answer = generate_answer(system_prompt, user_prompt)
#     return {"Final_answer" : answer}

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_rag(data: QueryRequest):
    if index is None:
        return {"Final_answer": "No document uploaded."}

    query_vector = embed_query(data.question)
    results = search(index, query_vector, chunks, top_k=3)
    context = build_context(results)
    system_prompt, user_prompt = build_rag_prompt(context, data.question)
    answer = generate_answer(system_prompt, user_prompt)

    return {"Final_answer": answer}


