import time
from langchain.chains.retrieval_qa.base import RetrievalQA
from sliding_window import *
import os


load_dotenv()

start_time = time.time()
wcs_cluster_url = os.getenv("WEAVIATE_URL")
wcs_api_key = os.getenv("WEAVIATE_API_KEY")

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcs_cluster_url,
    auth_credentials=weaviate.auth.AuthApiKey(api_key=wcs_api_key)
)

embedding_models = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


vectorstore = WeaviateVectorStore(
    client=weaviate_client,
    index_name="Chunks",
    text_key="text",
    embedding=embedding_models,
)


llm=ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Your task is to answer the user's question by utilizing the context provided.
                "When formulating your response, please use *only* the information found in the 'Context' section below. Ensure your answer is comprehensive, well-structured, and consists of a **minimum of two distinct paragraphs**. "
               "Please avoid providing short or superficial answers; instead, elaborate on the provided information in a detailed manner:\n\n{context}"""),
    ("human", "{question}")
])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    input_key="question"
)

question = "stresle nasıl başa çıkablirim?"
response = qa_chain.invoke({"question": question})

char_count=len(response)
total_token=char_count*0.7

end_time = time.time()
total_time = end_time - start_time

print(response)
print("total time:", total_time)
print("total token:", total_token)

