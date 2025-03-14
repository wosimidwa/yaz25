#RAG kullanarak belli bir alanda (yurtdışı) bilgilendirme sağlayabilen chabot kodu 

#%%openai API key
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    
#%% imports
import os
import openai
import chromadb
import requests
from bs4 import BeautifulSoup
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA

#%%
def scrape_website(url, question_tag, answer_tag):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data from {url}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    questions = soup.find_all(question_tag)
    answers = soup.find_all(answer_tag)
    
    data = []
    for q, a in zip(questions, answers):
        question = q.text.strip()
        answer = a.text.strip()
        data.append((question, answer))
    
    return data

def initialize_vector_db():
    client = chromadb.PersistentClient(path="db")
    vector_db = Chroma(client=client, embedding_function=OpenAIEmbeddings())
    return vector_db

def add_data_to_db(vector_db, data):
    docs = [Document(page_content=answer, metadata={"question": question}) for question, answer in data]
    vector_db.add_documents(docs)

def create_chatbot(vector_db):
    llm = ChatOpenAI(model_name="gpt-4")
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain

def main():
    vector_db = initialize_vector_db()
    
    sources = [
        ("https://satsuite.collegeboard.org/sat", "h3", "p"),
        ("https://www.ets.org/toefl", "h2", "p"),
        ("https://www.ucas.com/undergraduate", "h3", "p"),
        ("https://www.daad.de/en/study-and-research-in-germany/", "h3", "p"),
        ("https://www.educanada.ca/", "h3", "p"),
        ("https://www.studyaustralia.gov.au/", "h3", "p"),
        ("https://www.commonapp.org/", "h2", "p"),
        ("https://erasmus-plus.ec.europa.eu/", "h3", "p"),
        ("https://www.scholars4dev.com/", "h3", "p")
    ]
    
    all_data = []
    for url, q_tag, a_tag in sources:
        all_data.extend(scrape_website(url, q_tag, a_tag))
    
    if all_data:
        add_data_to_db(vector_db, all_data)
    
    chatbot = create_chatbot(vector_db)
    print("Chatbot başlatıldı! Çıkış için 'exit' yazın.")
    while True:
        query = input("Soru: ")
        if query.lower() == "exit":
            break
        response = chatbot.run(query)
        print("Yanıt:", response)

if __name__ == "__main__":
    main()
