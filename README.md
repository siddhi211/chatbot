# DialoGPT Chatbot

A simple interactive chatbot built using Microsoft's DialoGPT-medium model with a Gradio interface.

## Features
- Conversational AI powered by DialoGPT-medium
- User-friendly web interface using Gradio
- Real-time response generation
- Conversation memory (maintains context across messages)

## Prerequisites
- Python 3.7 or higher
- pip package manager

## Installation

1. Install the required packages:
```bash
pip install transformers
pip install gradio
pip install torch
```
## To run the chatbot
```bash
python chatbot.py
```
## Troubleshooting
Make sure PyTorch is installed correctly
```bash
python -c "import torch; print(torch.__version__)"
```
## Usage
The web interface will be available at http://localhost:7860 by default
Technology Used - Gradio, Hugging Face Transformers

![image](https://github.com/user-attachments/assets/24404f4c-0b0e-4cb0-a10c-c369ab6a0d33)
