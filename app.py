import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS #database to store vector in local machine (data will be erased everytime)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

#For venv - https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/
# guide: https://www.youtube.com/watch?v=dXxQ0LR-3Hg

def get_pdf_text(pdf_docs):
    text = "" 
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) #gives pages from which text is extracted using extract_text method
        for page in pdf_reader.pages:
            text += page.extract_text() #the text extracted from extract_text method is appended to text
    return text

def get_text_chunks(text):
    # https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/character_text_splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text) #split_text
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings() #free credits are exhausted! 
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") #download the 5GB bin file
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    #check this func out
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write("Done with Processing!")
                # create conversation chain
                #conversation = get_conversation_chain(vectorstore) 
                #to make the code persistent, st.session_state.conversation is used, as streamlit has the tendency to reload itself and evey change.
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()