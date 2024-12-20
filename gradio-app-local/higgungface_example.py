import os
from threading import Thread
from typing import Iterator

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


DESCRIPTION = """\
# Gemma 2 2B JPN IT

Gemma-2-JPN は Gemma 2 2B を日本語で fine-tune したものです。Gemma 2 の英語での性能と同レベルの性能で日本語をサポートします。

(Gemma-2-JPN is a Gemma 2 2B model fine-tuned on Japanese text. It supports the Japanese language at the same level of performance as English-only queries on Gemma 2.)
"""

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_id = "google/gemma-2-2b-jpn-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./local_model")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir="./local_model",
)
model.config.sliding_window = 4096
model.to(device)
model.eval()


@spaces.GPU
def generate(
    message: str,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = chat_history + [{"role": "user", "content": message}]

    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


demo = gr.ChatInterface(
    fn=generate,
    type="messages",
    description=DESCRIPTION,
    css_paths="style.css",
    fill_height=True,
    additional_inputs_accordion=gr.Accordion(label="詳細設定", open=False),
    additional_inputs=[
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["こんにちは、自己紹介をしてください。"],
        ["マシンラーニングについての詩を書いてください。"],
        [
            "次の文章を英語にして: Gemma-2-JPN は Gemma 2 2B を日本語で fine-tune したものです。Gemma 2 の英語での性能と同レベルの性能で日本語をサポートします。"
        ],
    ],
    cache_examples=False,
)


if __name__ == "__main__":
    demo.launch()