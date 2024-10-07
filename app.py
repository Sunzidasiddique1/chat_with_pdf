import os
import time
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

load_dotenv()
os.environ['GROQ_API_KEY'] = ['api_key']

# Function to load and process documents from a URL
def get_docs_from_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs)
    st.sidebar.write('Documents Loaded from URL:', len(split_docs))
    return split_docs

# Function to load and process documents from an uploaded PDF file
def get_docs(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(documents)
    os.remove("temp.pdf")  # Clean up the temporary file
    st.sidebar.write('Documents Loaded:', len(final_documents))
    return final_documents

# Function to create vector store
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True})
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.sidebar.write('Vector Store is ready with', len(docs), 'documents')
    return vectorstore

# Function to interact with Groq AI
def chat_groq(messages):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    response_content = ''
    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=1024,
        temperature=1.3,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += content
    return response_content

# Main function for Streamlit app
def main():
    st.set_page_config(page_title='DocuQuery')

    st.title("DocuQuery - Chatbot Interface")

    # Initialize session state variables
    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for document source selection
    st.sidebar.subheader("Choose document source:")
    option = st.sidebar.radio("Select one:", ("Upload PDF", "Enter Web URL"))

    if option == "Upload PDF":
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file is not None and st.session_state.docs is None:
            with st.spinner("Loading documents..."):
                st.session_state.docs = get_docs(uploaded_file)

    elif option == "Enter Web URL":
        url = st.sidebar.text_input("Enter URL", key="url_input")
        if url and st.sidebar.button('Process URL') and st.session_state.docs is None:
            with st.spinner("Fetching and processing documents..."):
                st.session_state.docs = get_docs_from_url(url)

    # Create Vector Store
    if st.session_state.docs is not None and st.sidebar.button('Create Vector Store'):
        with st.spinner("Creating vector store..."):
            st.session_state.vectorstore = create_vector_store(st.session_state.docs)

    # Chatbot UI
    if prompt := st.chat_input("Your question"):  # Capture user input
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Check for greeting and respond accordingly
    if st.session_state.vectorstore:
        if st.session_state.messages:
            user_message = st.session_state.messages[-1]["content"]
            # Chat interaction with document context
            def submit_with_doc():
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                context = retriever.invoke(user_message)
                
                # Debugging line
                st.sidebar.write('Retrieved Context:', context)
                
                prompt = f'''
                Based on the provided context, answer the user's question.

                Context: {context}

                Previous Questions and Answers: {st.session_state.chat_history}

                Latest Question: {user_message}
                '''
                messages = [{"role": "system", "content": "You are a helpful assistant"}]
                messages.append({"role": "user", "content": prompt})

                # Fetch response from Groq
                try:
                    ai_response = chat_groq(messages)
                except Exception as e:
                    st.error(f"Error during Groq interaction: {str(e)}")
                    ai_response = "An error occurred. Please try again."

                # Update chat history and display response
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

            submit_with_doc()
        else:
            st.warning("Please enter a question before submitting.")

    # Chat interaction without document context
    else:
        def submit_without_doc():
            if st.session_state.messages:  # Check if there are any messages
                user_message = st.session_state.messages[-1]["content"]
                prompt = f'''
                Answer the user's question based on the latest input in the chat.

                Previous Questions and Answers: {st.session_state.chat_history}

                Latest Question: {user_message}
                '''
                messages = [{"role": "system", "content": "You are a helpful assistant"}]
                messages.append({"role": "user", "content": prompt})

                try:
                    ai_response = chat_groq(messages)
                except Exception as e:
                    st.error(f"Error during Groq interaction: {str(e)}")
                    ai_response = "An error occurred. Please try again."

                # Update chat history and display response
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

            else:
                st.warning("Please enter a question before submitting.")

        submit_without_doc()

if __name__ == "__main__":
    main()
