import streamlit as st
import os
import time
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# 初始化文件夹
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('jj'):
    os.mkdir('jj')

# 初始化Streamlit会话状态
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

if 'vectorstore_initialized' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='jj',
                                          embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                              model="llama3"))
    st.session_state.vectorstore_initialized = True

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model="llama3",
                                  verbose=True,
                                  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("防灾大模型")


def initialize_static_knowledge():
    static_knowledge = [
        "你是防灾大模型，专门用来帮助人们理解和预防灾难。",
        "请始终使用中文回答用户的问题，确保信息准确传达。"
    ]
    # 确保使用正确的方法获取嵌入
    embedding_model = OllamaEmbeddings(base_url='http://localhost:11434', model="llama3")
    embeddings = [embedding_model.embed(text) for text in static_knowledge]
    for idx, embedding in enumerate(embeddings):
        st.session_state.vectorstore.add_document(embedding, metadata={"id": idx, "text": static_knowledge[idx]})
    st.session_state.vectorstore.persist()



# 在应用启动时初始化静态知识
if 'knowledge_initialized' not in st.session_state:
    initialize_static_knowledge()
    st.session_state.knowledge_initialized = True

uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    if not os.path.isfile(f"files/{uploaded_file.name}.pdf"):
        with st.status("正在读取你的文档..."):
            bytes_data = uploaded_file.read()
            with open(f"files/{uploaded_file.name}.pdf", "wb") as f:
                f.write(bytes_data)
            loader = PyPDFLoader(f"files/{uploaded_file.name}.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
            all_splits = text_splitter.split_documents(data)
            st.session_state.vectorstore = Chroma.from_documents(documents=all_splits,
                                                                 embedding=OllamaEmbeddings(model="nomic-embed-text"))
            st.session_state.vectorstore.persist()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    if user_input := st.chat_input("你:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("助手正在输入..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)
else:
    st.write("请上传PDF文件。")
