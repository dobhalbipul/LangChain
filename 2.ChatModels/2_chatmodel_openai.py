from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# temperature=0.0 - 0.3 means the model will be deterministic and predictable
# (i.e. it will give the same answer every time for the same question)
# whereas 0.7 -1.5 means the model will be more creative, random and diverse
# (i.e. it will give different answers for the same question)
# temperature=0.0 - 0.3 is good for factual questions
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
result = model.invoke("What is the capital of France?")

# OPENAI mode was giving a string but chat models are giving a dict
# so we need to access the content of the message
print(result.content)