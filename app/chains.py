import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser 
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        print('working')
        self.llm = ChatGroq(
            temperature = 0,
            groq_api_key = 'your_api_key',
            model_name = 'llama-3.3-70b-versatile' 
        )
    
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
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
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(
            input = {'page_data': cleaned_text}
        )
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
            return res
        except OutputParserException:
            raise OutputParserException('Contect is too big. Unable to parse job description.')
        return res if isinstance(res, list) else [res]
        
    def write(self, job, links):
        prompt_email = PromptTemplate.from_template(
            '''
            ### JOB DESCRIPTION:
            {job_description}
    
            ### INSTRUCTION:
            You are [name], a [role] at [company].
            Over your experience, you have empowered projects with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the company regarding the job mentioned above describing your capability
            in fulfilling their needs and expressing your interest in joining their company.
            Also add the most relevant ones from the following links to showcase your portfolio: {list_link}
            Remember you are [name], [role] at [company]. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            '''
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke(
            {'job_description': str(job), 'list_link': links}
        )
        return res.content
