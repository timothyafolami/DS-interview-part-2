import os
from pinecone import Pinecone
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

client = OpenAI()
# accessing the database
index_name = 'productrec'
pc = Pinecone()
pc_db = pc.Index(index_name)

# openai embedding
embedding = OpenAIEmbeddings()

# prompt
prompt = '''
    You are a Product Database Query Reviewer. 
    Your job is to review the queries passed in by the user to the database. 
    This is what you check for:
        1. Does Query include hateful word?
        2. Does the Query looks harmful or hateful.
    If the Query does not pass that requirement, your output is : "BAD QUERY".
    If the Query passed the requirement, you return back the Query as output.

    Please make sure to always give the right out as specified.

    Query:
    {query}

    Output:

'''

# second prompt

prompt_out = '''
    You are a Product Database Response. 
    Your job is to give a comment based on the output of database query. 
    The output is a list. 
    This is what you check for in the list:
        1. Does the ouput contains "BAD QUERY"?
    If the output conntains BAD QUERY; give a response to the user telling him he sent a bad query, 
    he should review and ask for something better. 
    
    If the output has real product name in it, then give short comments on each of the products.

    Please make sure to always give the right out as specified.

    Output:
    {output}

    Output:

'''

def query_reviewer(query):
    #Creating a dictionary to hold the query
    user_query = {'query': query}
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-16k",
    messages = [
    {
        "role": "system",
        "content": prompt.format(**user_query)
    }
],
    temperature=0,
    )
    return response.choices[0].message.content


# Adding the query reviewer to the product_query function
def product_query(query: str):
    # query review
    reviewed_query = query_reviewer(query)
    # condition based on the reviewed query
    if reviewed_query == 'BAD QUERY':
        return 'BAD QUERY'
    else:
        # embedding thr query
        query_embedding = embedding.embed_query(reviewed_query)
        # searching the database
        retrieved_docs = pc_db.query(
            vector = query_embedding,
            # returning the top 3 values
            top_k=5, 
            include_values=False,
            include_metadata=True,
        )
        # returning the reteieved docs
        similar_items = []
        for item in retrieved_docs["matches"]:
            text = item["metadata"]["text"]
            similar_items.append(text)
        return similar_items



def output_response(output):
    #Creating a dictionary to hold the query
    user_query_output = {'output': list(output)}
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-16k",
    messages = [
    {
        "role": "system",
        "content": prompt_out.format(**user_query_output)
    }
],
    temperature=0,
    )
    return response.choices[0].message.content
