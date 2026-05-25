import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# DeepSeek Setting
llm = ChatOpenAI(
    openai_api_key="sk-31df9e2d7db6401682aa3cc928808913",
    openai_api_base="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.7
)

prompt = PromptTemplate.from_template("你現在是「Healing Toys」電商平台的專業 AI 諮商師，你的任務是安撫學生並直接推薦玩偶。")
chain = prompt | llm

if __name__ == "__main__":
    result = chain.invoke({})
    print(result.content) # Display the response from the model