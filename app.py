import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from huggingface_hub import login
import os

# Load the Llama 3.2 model and tokenizer
model_name = 'meta-llama/Llama-3.2-1B'.strip()
access_token = os.getenv('HF_TOKEN')
st.write(f"Model name: '{model_name}' (type: {type(model_name)})")
st.write(f"access_token: '{access_token}' (type: {type(access_token)})")

if access_token is None:
    st.error("HF_TOKEN environment variable is not set.")
    st.stop()

login(token=access_token)

try:
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    st.write(f"(type: {type(tokenizer)})")
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

try:
    model = LlamaForCausalLM.from_pretrained(model_name)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Predefined FAQs
faq_data = {
    "What is your return policy?": "You can return any item within 30 days of purchase for a full refund.",
    "How can I track my order?": "You can track your order using the tracking link sent to your email.",
    "What payment methods do you accept?": "We accept credit cards, PayPal, and bank transfers.",
    "How do I contact customer support?": "You can contact customer support via email at support@example.com.",
}

def generate_response(user_message):
    # Check if the user message matches any FAQ
    for question, answer in faq_data.items():
        if user_message.lower() in question.lower():
            return answer
    
    # If no match found, use Llama 3.2 to generate a response
    inputs = tokenizer.encode(user_message, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Streamlit UI
st.title("FAQ Chatbot")
st.write("Ask me anything about our services!")

user_message = st.text_input("Your question:")

if user_message:
    bot_response = generate_response(user_message)

    # Fallback mechanism for unknown questions
    if bot_response not in faq_data.values():
        bot_response = "I'm sorry, I don't know the answer to that question. Can you please rephrase it?"

    st.write(f"**Bot:** {bot_response}")
    
'''
import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from huggingface_hub import login
import os

# Load the Llama 3.2 model and tokenizer
# Replace with the actual model path or name
model_name =  'meta-llama/Llama-3.2-1B'
access_token = os.getenv('HF_TOKEN')

login(token=access_token)

try:
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    ### st.stop()

model = LlamaForCausalLM.from_pretrained(model_name)

# Predefined FAQs,
faq_data = {
    "What is your return policy?": "You can return any item within 30 days of purchase for a full refund.",
    "How can I track my order?": "You can track your order using the tracking link sent to your email.",
    "What payment methods do you accept?": "We accept credit cards, PayPal, and bank transfers.",
    "How do I contact customer support?": "You can contact customer support via email at support@example.com.",
}

def generate_response(user_message):
    # Check if the user message matches any FAQ
    for question, answer in faq_data.items():
        if user_message.lower() in question.lower():
            return answer
    
    # If no match found, use Llama 3.2 to generate a response
    inputs = tokenizer.encode(user_message, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Streamlit UI
st.title("FAQ Chatbot")
st.write("Ask me anything about our services!")

user_message = st.text_input("Your question:")

if user_message:
    bot_response = generate_response(user_message)

    # Fallback mechanism for unknown questions
    if bot_response not in faq_data.values():
        bot_response = "I'm sorry, I don't know the answer to that question. Can you please rephrase it?"

    st.write(f"**Bot:** {bot_response}")
'''