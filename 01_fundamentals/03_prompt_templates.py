import os 
from dotenv import load_dotenv 
from langchain_core.prompts import ChatPromptTemplate #this is so that we can use the prompt template class to create a prompt template
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b") #this is the model we are going to use, we can change this to any of the models we want to test.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant English to {language}."),
    ("human", "{text}")
])

filled_prompt = prompt_template.format(language="Arabic", text="I love programming") #this is where we fill in the prompt template with the values we want to use. In this case, we are telling the model to translate the text "I love programming" to Arabic.

ai_msg = llm.invoke(filled_prompt) #this is where we send the filled prompt to the model to get a response.
print(ai_msg.content) #this is where we print the response from the model. We expect the model to return "J'aime la programmation", which is the French translation of "I love programming".