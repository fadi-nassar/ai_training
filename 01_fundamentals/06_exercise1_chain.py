import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to {language}."),
    ("human", "{text}")
])

parser = StrOutputParser()

models_to_test = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

test_cases = [
    {"language": "French", "text": "I love programming."},
    {"language": "Spanish", "text": "The weather is nice today."},
    {"language": "Arabic", "text": "Where is the nearest hospital?"},
]

for model_name in models_to_test:
    llm = ChatGroq(model=model_name)
    chain = prompt_template | llm | parser

    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print('='*50)

    for case in test_cases:
        result = chain.invoke(case)
        print(f"[{case['language']}] {case['text']} -> {result}")