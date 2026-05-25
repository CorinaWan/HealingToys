from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class InputModel(BaseModel):
    input: str

llm = ChatOpenAI(
    openai_api_key="sk-31df9e2d7db6401682aa3cc928808913", 
    openai_api_base="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.7
)

prompt = PromptTemplate.from_template("你是一个有用的助手，请回答用户的问题：{input}")

chain = prompt | llm | StrOutputParser()

rag_mongo_chain = chain.with_types(input_type=InputModel, output_type=str)