import os 
import google.generativeai as genai
from PDF_Extractor import text_extractor
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# Lets configure the model

#LLM Model

gemini_key = os.getenv('GOOGLE_API_KEY1')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')


# Configure Embedding ModuleNotFoundError

embedding_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')

# Lets create the main page

st.title(':orange[CHATBOT:] :blue[AI Assisted chatbot using RAG]')
tips = '''
Follow the steps to use the application:-
1. Upload your PDF Document in sidebar
2. Write a query and start the chat.
'''
st.text(tips)

# Lets create the sidebar

st.sidebar.title(':rainbow[UPLOAD YOUR FILE]')
st.sidebar.subheader(' :green[Upload PDF File only.]')
pdf_file = st.sidebar.file_uploader('Upload here', type=['pdf'])

if pdf_file:
    st.sidebar.success('File Upload Successfully')
    
    file_text = text_extractor(pdf_file)
    
    # Step 1: Chunking 
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(file_text)
    
    # Step 2: Create the vector data base (FAISS)
    vector_store = FAISS.from_texts(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={'k':3})
    
    def generate_content(query):
        # Step 3: Retriever (R)
        retrived_docs = retriever.invoke(query)
        context = '\n'.join([d.page_content for d in retrived_docs])
        
        
        # Step 4: Augmenting (A)
        
        augmented_prompt = f'''
        <Role> You are a helpful assistant using RAG
        <Goal> Answer the question asked by the user. Here is the question: {query}
        <Context> Here are the documents retrived from the vector databse to support the answer which you have to generate: {context} '''
        
        # Step 5: Generate (G)
        
        response = model.generate_content(augmented_prompt)
        return response.text
    
    
    # Create chatbot in order to start the conversation 
    # Initialize a chat create history if not created 
    
    if 'history' not in st.session_state:
        st.session_state.history = []
        
        
    # Display the History 
    
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f':green[USER:] :blue[{msg['text']}]')
        else:
            st.write(f':red[CHATBOT:] :blue[{msg['text']}]')
            
            
    # Input from the user using streamlit form 
    
    with st.form('Chatbot Form', clear_on_submit=True):
        user_query = st.text_area('Ask Anything: ')
        send = st.form_submit_button('Send')
        
    # Start the conversation and append the output and query in history )    
    if user_query and send:
        st.session_state.history.append({'role':'user', 'text':user_query})
        st.session_state.history.append({'role':'chatbox', 'text':generate_content(user_query)})
        st.rerun()