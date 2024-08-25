import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def greet(text):
    return classifier(text)

demo = gr.Interface(
    fn=greet,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()
