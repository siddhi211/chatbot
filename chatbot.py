from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize conversation history
class Chatbot:
    def __init__(self):
        self.chat_history_ids = None
    
    def generate_response(self, user_input):
        # Encode user input
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Append to chat history or start new chat
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate response
        self.chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        
        # Get response text
        response = tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

# Create chatbot instance
chatbot = Chatbot()

# Create Gradio interface
interface = gr.Interface(
    fn=chatbot.generate_response,
    inputs="text",
    outputs="text",
    title="DialoGPT Chatbot with Memory",
    description="This chatbot remembers the conversation history"
)

interface.launch()