from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
import requests
from flask import Flask, request, jsonify
from pydantic import BaseModel
from dotenv import load_dotenv
from queue import Queue
from typing import List
app = Flask(__name__)
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text
# def get_pdf_text(pdf_docs):
#     text = ""
#     for txt_file in pdf_docs:
#         text += txt_file.read().decode('utf-8')
#         return text
load_dotenv()
task_queue = Queue()
class GetQuestionAndFactsResponse(BaseModel):
    question: str
    facts: List[str]
    status: str

def get_pdf_text(pdf_docs):
    text = ""
    for text_bytes in pdf_docs:
        # Reset the file pointer to the beginning in case it's not at the start
        # Read the contents of the file and decode from bytes to string
        text += text_bytes.read().decode('utf-8')
        return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


@app.route('/submit_question_and_documents', methods=['POST'])
def process_payload():
    # Parse the JSON payload
    data = request.get_json()
    if 'question' not in data or 'documents' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    question = data['question']
    text_files = []
    # Download PDFs from URLs
    for url in data['documents']:   
        print(url)
        response = requests.get(url)
        if response.status_code == 200: 
            text_files.append(response.content)
            print(response.content)
        else:
            return jsonify({"error": f"Failed to download file from {url}"}), 500
    # Extract text from PDFs
    # extracted_text = get_pdf_text(pdf_files)
    task_queue.put((text_files, question))
    # # Split text into chunks
    # text_chunks = get_text_chunks(extracted_text)

    # # Optional: Process text chunks further, e.g., with NLP models or vector stores
    # # Here you might use get_vectorstore and get_conversation_chain functions
    
    # return jsonify({"message": "Data processed successfully", "extractedText": extracted_text}), 200
    # response_data = {"question": question, "responses": responses}
    # session['recent_response'] = response_data  # Store in session for retrieval
    return jsonify({"status": "processing"}), 200

def process_documents(document_contents, question):
    responses = []

    for text_file in document_contents:
        print(text_file)
        extracted_text = str(text_file)
        text_chunks = get_text_chunks(extracted_text)
        vector_store = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vector_store)

            # Assuming we want to store the results of the conversation chain
        response_to_quesion = conversation_chain({'query':question})
        response_to_question = response_to_quesion['result']
        response = {"answer":response_to_question}
        responses.append(response)
    return responses

@app.route('/get_question_and_facts', methods=['GET'])
def get_facts():
    if not task_queue.empty():
        text_files, question = task_queue.get()
        print(len(text_files))
        responses = process_documents(document_contents=text_files, question=question)
        facts = [response['answer'] for response in responses]
        response_data = GetQuestionAndFactsResponse(facts=facts, question=question, status="done")
        return jsonify(response_data.dict()), 200
    else:
        return jsonify({"error": "No tasks in queue"}), 404

if __name__ == '__main__':
    app.run(debug=True)
