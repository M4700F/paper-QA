'''
1. take a user question
2. embed it with the same embedding model
3. search the vector db and retrieve the corresponding chunks
4. send the chunks along side with the question as a prompt
5. send this prompt to the LLM
6. return the grounded answer
'''
import os
import chromadb
from embed import embed
from google import genai
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = chromadb.PersistentClient(path="../data/chroma_db")
collection = client.get_or_create_collection("papers")

# gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

def retrieve_chunks(query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )
    return results["documents"][0]

def build_prompt(query, chunks):
    context = "\n\n".join(chunks)
    return f"""Answer the question using only the context below. If the answer isn't in the context, say you don't know.

    Context:
    {context}

    Question: {query}

    Answer:"""
    
def ask_llm(prompt):
    
    # response = gemini_client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents=prompt,
    # )
    # return response.text
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    query = "What method does this paper propose?"
    
    query_embedding = embed([query])
    # print(f"Query embedding shape: {query_embedding.shape}")
    
    chunks = retrieve_chunks(query_embedding=query_embedding)
    # print(f"\nRetrieved {len(chunks)} chunks:\n")
    # for i, chunk in enumerate(chunks):
    #     print(f"--- Chunk {i+1} ---")
    #     print(chunk[:300])
    #     print()
    
    prompt = build_prompt(query=query, chunks=chunks)
    answer = ask_llm(prompt=prompt)
    
    print(f"Question: {query}\n")
    print(f"Answer: {answer}")
    