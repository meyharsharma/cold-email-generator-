#chat groq is a class that langchain offers - getting fimilar with langchain_groq

from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature = 0,
    groq_api_key = 'your-api-key',
    model_name = 'llama-3.3-70b-versatile' #model will have to be updated as newer versions are released
)

#for demonstaration - ask a ques and store it in the response variable
response = llm.invoke('the first person to land on the moon was')
print(response.content)
