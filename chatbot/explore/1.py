from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.callbacks import get_openai_callback

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

chat_model = ChatOpenAI(model="gpt-4", temperature=0.0)

conversation_buf = ConversationChain(llm=chat_model, memory=ConversationBufferMemory())

res = conversation_buf.invoke("Good morning AI!")
res1 = count_tokens(conversation_buf, "Good morning AI!")

#res = conversation_buf.invoke("I want to learn more about AI.")
count_tokens(conversation_buf, "My interest here is to explore the potential of integrating Large Language Models with external knowledge")
count_tokens(conversation_buf, "Which data source types could be used to train a model for a specific task?")
print(conversation_buf.memory.buffer_as_str)


## Conversation Buffer Memory

conversation_buf = ConversationChain(llm=chat_model, memory=ConversationBufferMemory())
res = conversation_buf.invoke("Good morning AI!")
print(conversation_buf.memory.buffer_as_str)
res = conversation_buf.invoke("I want to learn more about AI.")
print(conversation_buf.memory.buffer_as_str)
res = conversation_buf.invoke("What is the best way to learn about AI?")
print(conversation_buf.memory.buffer_as_str)

