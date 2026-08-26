import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = chat_groq = ChatGroq(model="openai/gpt-oss-120b") 
Parser = StrOutputParser() # This is a parser that will parse the output of the model into a JSON object. This is useful for extracting structured data from the model's output.

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to {language}."),
    ("human", "{text}")
])

chain = prompt_template | llm | Parser # This creates a chain that first formats the prompt and then sends it to the model.

french_result = chain.invoke({"language": "French", "text": "I love programming."})
arabic_result = chain.invoke({"language": "Arabic", "text": "I love programming."})
print(french_result)

print(arabic_result)
