from transformers import pipeline
import gradio as gr

# Use a pre-trained text generation model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Create interface with Gradio
def generate_response(text):
    response = chatbot(text, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

interface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="My First Chatbot"
)

interface.launch()