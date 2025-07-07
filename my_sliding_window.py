import os
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import weaviate
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore

load_dotenv()
wcs_cluster_url = os.getenv("WEAVIATE_URL")
wcs_api_key = os.getenv("WEAVIATE_API_KEY")

doc_loader = PyPDFLoader("yourDocuments.pdf")
pdf_page = doc_loader.load()
text = "\n".join(doc.page_content for doc in pdf_page)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700)
pages = text_splitter.split_text(text)


class Chunk(BaseModel):
    """Represents a single meaningful chunk of text from the document."""
    title: str = Field(description="A concise title summarizing the main topic of this chunk.")
    content: str = Field(description="The full text content of the chunk.")


class DocumentChunks(BaseModel):
    """Represents the complete set of chunks extracted from the document."""
    chunks: List[Chunk] = Field(description="A list of meaningful and semantically coherent chunks from the document.")


def sliding_window_chunking(pages: List[str]) -> List[Chunk]:
    parser = PydanticOutputParser(pydantic_object=DocumentChunks)
    format_instructions = parser.get_format_instructions()

    PROMPT_TEMPLATE = """
    You are an AI assistant specialized in **deep document understanding** and **semantic chunking**—intelligently breaking down text into coherent and meaningful units.
    
    ### YOUR TASK:
    Using the provided context (`Previous Page`, `Target Page`, and `Next Page`), your job is to split the `Target Page` into **logically and semantically coherent chunks**.
    
    Follow the rules below carefully when performing the chunking:
    
    ---
    
    ###RULES
    
    **1. Semantic Coherence (Priority #1):**  
    Each chunk must be a **logically complete and meaningful unit on its own**.  
    - Never split a sentence, paragraph, or logical argument in the middle.  
    - Do not start a new chunk until the current topic or subtopic is fully concluded.
    
    **2. Chunk Length (Target):**  
    - Each chunk should aim to be **approximately 1000 characters long**.  
    - This is a guideline, not a strict rule.  
      (For example: A coherent section ending at 800 characters is acceptable.)
    
    **3. Use of Context (Cross-Page Reasoning):**  
    Leverage `Previous Page` and `Next Page` to ensure complete and accurate chunks.  
    - If the `Target Page` starts mid-sentence or mid-paragraph, include the missing beginning from the `Previous Page`.  
    - If a topic on the `Target Page` continues on the `Next Page`, include enough from the `Next Page` to complete that topic in the final chunk.
    
    **4. Natural Break Points:**  
    Prioritize natural division points such as:  
    - Headings and subheadings  
    - Numbered or bulleted lists  
    - Clear thematic transitions
    
    **5. Output Format (Strict Requirement):**  
    Your output must strictly follow the **JSON format** provided below:
    
   {format_instructions}
    """

    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash",
                                 temperature=0.1)  # gpt-4o genellikle daha iyi formatlama yapar

    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATE),
        ("human",
         "Lütfen sağlanan metinleri analiz et ve görevi tamamla:\n\n[Önceki Sayfa]:\n{previous_page}\n\n[Hedef Sayfa]:\n{target_page}\n\n[Sonraki Sayfa]:\n{next_page}")
    ])

    chain = prompt | llm | parser

    all_chunks = []

    for i in range(len(pages)):
        if i == 0:
            previous_page = ""
        else:
            previous_page = pages[i - 1]

        target_page = pages[i]

        next_page = ""
        if i + 1 != len(pages):
            next_page = pages[i + 1]

        try:
            result: DocumentChunks = chain.invoke({
                "previous_page": previous_page,
                "target_page": target_page,
                "next_page": next_page,
                "format_instructions": format_instructions

            })
            print(len(result.chunks))
            all_chunks.extend(result.chunks)

        except Exception as e:
            print(f"Hata oluştu (sayfa {i + 1}): {e}")

    return all_chunks


client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcs_cluster_url,
    auth_credentials=weaviate.auth.AuthApiKey(api_key=wcs_api_key)
)
if client.collections.exists("Chunks") == False:
    chunks = sliding_window_chunking(pages)
    embedding_s = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    documents = [
        Document(page_content=chunk.content, metadata={"title": chunk.title})
        for chunk in chunks
    ]

    vectorstore = WeaviateVectorStore.from_documents(
        documents=documents,
        embedding=embedding_s,
        client=client,
        index_name="SlidingChunks",
    )


