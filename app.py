# importing dependencies
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


# creating custom template to guide llm model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# extracting text from pdf
def get_pdf_text(docs):
    text=""
    for pdf in docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# converting text to chunks
def get_chunks(raw_text, pdf_name):
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len)   
    chunks=text_splitter.split_text(raw_text)
    # Add metadata to chunks
    return [{"text": chunk, "source": pdf_name} for chunk in chunks]

# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(text_chunks):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [{"source": chunk["source"]} for chunk in text_chunks]
    vectorstore=faiss.FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vectorstore

# generating conversation chain  
def get_conversationchain(vectorstore):
    llm=ChatOpenAI(temperature=0.2)
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory)
    return conversation_chain

# generating response from user queries and displaying them accordingly
def handle_question(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response["chat_history"]
    
    # Get source documents
    docs = st.session_state.vectorstore.similarity_search(question, k=3)
    sources = set(doc.metadata["source"] for doc in docs)
    
    answer = response["chat_history"][-1].content
    
    # Display message pair
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        st.write(answer)
        
        # Display sources
        if sources:
            st.write("Sources:", ", ".join(sources))

def main():
    load_dotenv()
    st.set_page_config(page_title="Patent Knowledge Base Assistant", page_icon="📚", layout="wide")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.header("Patent Knowledge Base Assistant 📚")

    # Sidebar for document upload
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                text_chunks = []
                for doc in docs:
                    raw_text = get_pdf_text([doc])
                    chunks = get_chunks(raw_text, doc.name)
                    text_chunks.extend(chunks)
                
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversationchain(vectorstore)
    
    # Display chat history
    if st.session_state.chat_history:
        for i in range(0, len(st.session_state.chat_history), 2):
            with st.chat_message("user"):
                st.write(st.session_state.chat_history[i].content)
            with st.chat_message("assistant"):
                st.write(st.session_state.chat_history[i+1].content)
    
    # Chat input
    if question := st.chat_input("Ask a question about your documents:"):
        if st.session_state.conversation is None:
            st.error("Please upload and process documents first!")
        else:
            handle_question(question)

if __name__ == '__main__':
    main()