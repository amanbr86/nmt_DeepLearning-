"""Gradio UI.  Run from the project root:  python -m app.gradio_app"""
import os

import gradio as gr

from src.inference import Translator

CKPT_DIR = os.getenv("NMT_CKPT_DIR", "checkpoints")
translators = {}
for name in ["seq2seq", "attention"]:
    if os.path.exists(f"{CKPT_DIR}/{name}_best.pt"):
        translators[name] = Translator(name, CKPT_DIR)


def run(text, model):
    if not text.strip():
        return ""
    return translators[model].translate(text)


demo = gr.Interface(
    fn=run,
    inputs=[gr.Textbox(lines=3, label="English", placeholder="I am going to the university."),
            gr.Radio(list(translators), value="attention" if "attention" in translators else None, label="Model")],
    outputs=gr.Textbox(label="Amharic"),
    title="English → Amharic Neural Machine Translation",
    description="Seq2Seq LSTM vs Attention LSTM trained from scratch.",
    examples=[["I am going to the university.", "attention"], ["Where do you live?", "attention"]])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
