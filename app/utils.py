import re

def clean_text(text):
    #remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    #remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    #remove special characters 
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    #replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    #trim leading and trailing whitespace
    text = text.strip()
    #remove extra whitespace
    text = ' '.join(text.split())
    return text