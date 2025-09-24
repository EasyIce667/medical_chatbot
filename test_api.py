from huggingface_hub import InferenceClient
client = InferenceClient(token="your_new_token_here")
try:
    response = client.text_generation("Test prompt", model="mistralai/Mistral-7B-Instruct-v0.3")
    print(response)
except Exception as e:
    print(f"API test failed: {e}")