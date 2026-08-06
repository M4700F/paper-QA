import pymupdf

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text(pdf_path):
    '''
    this function extracts text from pdf
    returns the whole pdf's text
    '''
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=150):
    '''
    arg: text, chunk_size, overlap
    return a list of chunks
    '''
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed(chunks):
    return model.encode(chunks)
    
if __name__ == "__main__":
    text = extract_text("../papers/perpetual_wonder.pdf")
    chunks = chunk_text(text)
    
    # print(chunks[0:10])
    print(f"Total characters: {len(text)}")
    print(f"Number of chunks: {len(chunks)}")
    
    embeddings = embed(chunks)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First embedding (first 10 values): {embeddings[0][:10]}")
    