#!/usr/bin/env python
# coding: utf-8

# In[2]:


from langchain_groq import ChatGroq
llm  = ChatGroq(
    temperature = 0,
    groq_api_key = 'your-api-key',
    model_name = 'llama-3.3-70b-versatile' 
)
response = llm.invoke('The first person to land on the moon was') #checking if model is operational
print(response.content)


# In[10]:


#using langchain's webscrapper to get nike's Lead Software Engineer role scrapped
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader('https://careers.nike.com/lead-software-engineer/job/R-55111')
page_data = loader.load().pop().page_content 
print(page_data)


# In[11]:


from langchain_core.prompts import PromptTemplate
prompt_template= PromptTemplate.from_template(
    '''
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    
    ### INSTRUCTION:
    The scraped text is from the careers page of a website.
    Extract all job postings and return them in a structured JSON format with the following keys:
    - "role": The job title.
    - "experience": Required years or level of experience.
    - "skills": A list of required skills.
    - "description": A brief summary of the job.

    Only return the valid JSON. Do not include any preamble or explanations.
    
    ### OUTPUT:
    '''
)
#pipe operator to pass the prompt template with the llm
chain = prompt_template | llm
res = chain.invoke(input = {'page_data':page_data})
print(res.content)


# In[13]:


type(res.content)


# In[20]:


#converting file to a json type
from langchain_core.output_parsers import JsonOutputParser 
json_parser = JsonOutputParser()
json_res = json_parser.parse(res.content)
json_res


# In[21]:


type(json_res)


# In[22]:


#setting up chromadb 
#dummy data file 
import pandas as pd 
df = pd.read_csv('/Users/meyhar/Documents/learning dump/cold email project/my_portfolio.csv')
df


# In[27]:


#downloading data to the vector database 
#persitent client allows for vector db to store on disk and not memory 
import chromadb
import uuid
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name = 'portfolio')

if not collection.count():
    for _,row in df.iterrows():
        collection.add(documents = row['Techstack'],
                       metadatas = {'links': row['Links']},
                       ids = [str(uuid.uuid4())])


# In[28]:


#requesting two results and metadata which are the links for the relevant skills
links = collection.query(query_texts = ['Experience in Python', 'Expertise in React'], n_results = 2).get('metadatas', [])
links


# In[31]:


job = json_res[0]
job['skills']


# In[32]:


links = collection.query(query_texts = job['skills'], n_results = 2).get('metadatas', [])
links


# In[36]:


prompt_email = PromptTemplate.from_template(
     """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Rachel, a Software Engineer at Samsung.
        Over your experience, you have empowered projects with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the company regarding the job mentioned above describing your capability
        in fulfilling their needs and expressing your interest in joining their company.
        Also add the most relevant ones from the following links to showcase your portfolio: {list_link}
        Remember you are Rachel, SDE II at Samsung. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
)
chain_email = prompt_email | llm
res = chain_email.invoke({'job_description': str(job), 'list_link': links})
print(res.content)


# In[ ]:




