import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = chat_groq = ChatGroq(model="openai/gpt-oss-120b")
Parser = StrOutputParser() 

promt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to {language}."),
    ("human", "{text}")
])

chain_no_parser = promt_template | llm # This creates a chain that first formats the prompt and then sends it to the model.

chain_with_parser = promt_template | llm | Parser # This creates a chain that first formats the prompt and then sends it to the model, and then parses the output of the model into a JSON object.

print("--- WITHOUT parser ---")
result_raw = chain_no_parser.invoke({"language": "French", "text": "I love programming."})
print(result_raw)
print(type(result_raw))  # shows you what kind of object this is

print("\n--- WITH parser ---")
result_clean = chain_with_parser.invoke({"language": "French", "text": "I love programming."})
print(result_clean)
print(type(result_clean))  # shows you what kind of object this is