import gradio as gr
from chatbot_engine_agent import chat, create_index, setup_model_and_tokenizer, setup_conversation_agent, setup_pipeline
from dotenv import load_dotenv
import os

from langchain.memory import ChatMessageHistory

load_dotenv()
index = create_index()
model, tokenizer = setup_model_and_tokenizer()
llm = setup_pipeline(model, tokenizer)
agent = setup_conversation_agent(llm, index)

def response(message, chat_history):
    history = ChatMessageHistory()
    for msg in chat_history:
        if msg["role"] == "user":
            history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            history.add_ai_message(msg["content"])

    bot_message = chat(message, history, agent)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})

    return "", chat_history

def main():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type='messages')

        msg = gr.Textbox()
        clear = gr.Button("Clear")
        
        msg.submit(response, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch()

if __name__ == "__main__":
    main()
