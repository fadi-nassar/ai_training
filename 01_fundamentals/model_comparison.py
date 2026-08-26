import os # this is used to call the operating system to get the enviroment variable for the api key
from dotenv import load_dotenv # this is used to read the enviroment variable from a .env file
from langchain_groq import ChatGroq # this is the langchain wrapper class that knows how to talk to Groq's API. 
# what is an sdk? an sdk is a software development kit, which is a set of tools that allows developers to create applications for a specific platform. in this case, the langchain wrapper class is an sdk for Groq's API, which makes it easier to use in a conversational context.

load_dotenv() # this is what reads the env file and get the needed variables into this file/enviroment

models_to_test = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]
#this is a list of tghe models we want to test the bigger the number of parameters the more powerful the model is, but also the more expensive it is to use. 

messages= [
    ("system","you are a helpful assistant English to French. Translate the user sentence."),
    ("human","i love programming")
]

for model_name in models_to_test:
    llm = ChatGroq(model=model_name) # Initialize the ChatGroq instance with the current model
    ai_msg = llm.invoke(messages) # invoke the model with the message 
    print("\n--- Response ---")
    print(ai_msg.content)

    print("\n--- Stats ---")
    usage = ai_msg.response_metadata["token_usage"]
    print(f"Model:      {ai_msg.response_metadata['model_name']}")
    print(f"Tokens in:  {usage['prompt_tokens']}")
    print(f"Tokens out: {usage['completion_tokens']}")
    print(f"Total time: {usage['total_time']:.3f}s")