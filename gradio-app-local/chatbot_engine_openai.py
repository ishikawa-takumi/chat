from typing import List

import langchain
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

langchain.verbose = True


def create_index() -> VectorStoreIndexWrapper:
    # loader = DirectoryLoader("./src/", glob="**/*.py")
    loader = DirectoryLoader("./data/", glob="**/*.txt")
    embeddings = OpenAIEmbeddings() 
    index = VectorstoreIndexCreator(embedding=embeddings, vectorstore_cls=Chroma).from_loaders([loader])
    return index


def create_tools(index: VectorStoreIndexWrapper, llm) -> List[BaseTool]:
    vectorstore_info = VectorStoreInfo(
        vectorstore=index.vectorstore,
        name="udemy-langchain source code",
        description="Source code of application named udemy-langchain",
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    return toolkit.get_tools()

def setup_conversation_agent(index: VectorStoreIndexWrapper) -> ChatOpenAI:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    tools = create_tools(index, llm)

    agent_chain = initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, handle_parsing_errors=True # 再試行回数を制限

    )

    return agent_chain


def chat(
    message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper
) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    tools = create_tools(index, llm)

    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )

    agent_chain = initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory
    )

    return agent_chain.run(input=message)