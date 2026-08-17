import os #Python's built-in module for talking to the operating system. We're using it just to read environment variables
from dotenv import load_dotenv #A third-party module for reading environment variables from a .env file. This is useful for keeping sensitive information like API keys out of your codebase.
from langchain_groq import ChatGroq #the LangChain wrapper class that knows how to talk to Groq's API. Think of it like an SDK client for Groq's API, but with some extra features that make it easier to use in a conversational context.

load_dotenv() #this is what reads the env file and get the needed variables into this file/enviroment 

key = os.getenv("GROQ_API_KEY") #this is how we get the api key from the available enviroment variables.
print("key loaded from env file: ", key is not None, "-start with", key[:5] if key else "N\A") #just a quick check to make sure we got the key. We don't want to print the whole key, so we just print the first 5 characters.

llm = ChatGroq(model="llama-3.3-70b-versatile")
messages = ["system: You are a helpful assistant English to French. Translate the user sentence.", "user: i love programming"]
#here we chose what llm to use, and setup a conversation between the a user and assistan the system message is a prompt that tells the assistant what to do, and the user message is the input we want to translate.

ai_msg = llm.invoke(messages) #here is where the message will get sent to the model  to be use kind of like a function call, the model will return a respoce that we will compare to what we expected in this case we expect the model to return j'aime la programmation, which is the french translation of the user message.
print("\n--- Response ---")
print(ai_msg.content)

print("\n--- Stats ---")
usage = ai_msg.response_metadata["token_usage"]
print(f"Model:      {ai_msg.response_metadata['model_name']}")
print(f"Tokens in:  {usage['prompt_tokens']}")
print(f"Tokens out: {usage['completion_tokens']}")
print(f"Total time: {usage['total_time']:.3f}s")