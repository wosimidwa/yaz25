'''
to do 
kaynakça genişletilmesi
türkçe dil desteği
web site entegrasyonu

'''


import os
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Web scraping fonksiyonu
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
        question = q.get_text(strip=True)
        answer = a.get_text(strip=True)
        data.append(Document(page_content=answer, metadata={"question": question}))
    
    return data

def initialize_vector_db():
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="db", embedding_function=embedding_function)
    return vector_db

def add_data_to_db(vector_db, data):
    if data:
        vector_db.add_documents(data)

def create_chatbot(vector_db):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0.7,
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1"
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
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
    print("Chatbot başlatıldı!")
    while True:
        query = input("Soru: ")
        if query.lower() == "exit":
            break
        response = chatbot.invoke(query)
        print("Yanıt:", response["result"]) 

if __name__ == "__main__":
    main()

'''
türkçe dil desteği embedding:

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embedding boyutu değişimine dikkat et duruma göre reset at
'''
