import os
import gradio as gr
from chatbot_engine import setup_model_and_tokenizer, setup_pipeline, setup_vectorstore, chat_with_rag
from langchain.memory import ChatMessageHistory

# Set environment variable to disable upper limit for memory allocations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

model, tokenizer = setup_model_and_tokenizer()
llm = setup_pipeline(model, tokenizer)
vectorstore = setup_vectorstore("data")
# index = setup_index()

def response(message, chat_history):
    history = ChatMessageHistory()
    print("********")
    for msg in chat_history:
        if msg["role"] == "user":
            history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            history.add_ai_message(msg["content"])

    # bot_message = chat(llm, message, history)
    bot_message = chat_with_rag(llm, message, history, vectorstore)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})

    print("******************************")
    print(chat_history)
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