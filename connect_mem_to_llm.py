import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Setup LLM
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    chat_llm = ChatHuggingFace(llm=llm)  # Wrap for conversational task
    return chat_llm

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_SYSTEM_PROMPT = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.
Start the answer directly. No small talk please.
"""

CUSTOM_HUMAN_PROMPT = """
Context: {context}
Question: {question}
"""

def set_custom_prompt():
    # Fixed: Ensure input_variables includes 'context' and 'question'
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(CUSTOM_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(CUSTOM_HUMAN_PROMPT)
    ])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt()}
)

# Now invoke with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])