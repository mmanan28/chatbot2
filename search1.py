import openai
import langchain
import pinecone 
import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, PineconeConfigurationError, ServerlessSpec
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langchain.vectorstores import Pinecone
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from pinecone import Pinecone as pc
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI


from dotenv import load_dotenv
load_dotenv(dotenv_path=".env",override=True)

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

doc=read_doc('documents/')
len(doc)

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

documents=chunk_data(docs=doc)
len(documents)

embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
embeddings

vectors=embeddings.embed_query("How are you?")
len(vectors)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#print("hii ksie hoo" , OPENAI_API_KEY)

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable.")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize Pinecone with API key

#pinecone.init(api_key=PINECONE_API_KEY)

try:
    pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
except PineconeConfigurationError as e:
    print(f"Error initializing Pinecone: {e}")

index_name = "m4"

# Create a Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimensions=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'),
        host='https://m4-p23bwks.svc.aped-4627-b74a.pinecone.io'
    )

#index = LangchainPinecone.from_documents(documents, embeddings, index_name=index_name)

'''def retrieve_query(query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results

llm=OpenAI(model_name="text-embedding-3-small",temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")

def retrieve_answers(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response'''


text_field = "text"  # the metadata field that contains our text
# initialize the vector store object
vectorstore = Pinecone(
    pc.Index("m4"), embeddings.embed_query, text_field
)


greetings = ["hi", "helo", "hello", "hey", "good morning", "good evening","good evening","anyone here",   'howdy','salutations','hiya','hey there','good day',"what's up",'how are you','yo','hi there',"how's it going","how's everything","how's life",'nice to see you','pleased to meet you','good to see you','welcome','hi ya','hello there','how have you been',"what's going on",'how are things','how do you do',"what's happening","how's your day","how's your day going","what's new","what's good","how's it hanging",'how are things going',"how's your morning","how's your afternoon","how's your evening",'what have you been up to',
    'long time no see',"it's been a while",'how have you been lately',"how's everything going","how's it been","how's your week","what's the latest","how's your weekend",'what have you been doing',"what's been happening",'yes','no','sure','yeah','nope','yep','nah','okay','alright','absolutely','of course','definitely','affirmative','negative','indeed','certainly','sure thing','yup','uh huh','no way','not at all','by all means','no thanks','roger','right','fine','okay dokey','okie dokie','for sure','no problem','you bet','absolutely not','no doubt','unquestionably','without a doubt','no chance','yes please','not really','totally',"I'm in",'I agree',"I'm on board",'that works',"I'll pass",'not interested','why not','sure thing','count me in','you got it']
reply= [
    """Welcome to Iotric, what will you like to opt for.
    1. Explore our services.
    2. Explore our Portfolio.
    3. Iotric Products.
    4. Iotric Blogs.
    5. Contact us.
    6. Schedule a meeting."""
]

blog_keywords = ["blog", "blogs", "articles", "iotric blogs", "read", "latest blog", "read blog",
    "blog posts", "recent blogs", "recent articles", "write-ups", "latest articles",
    "blog section", "blog page", "blog content", "published articles", "featured blog",
    "latest write-ups", "blog updates", "new blogs", "new articles", "recent posts",
    "insights", "company blog", "industry articles", "tech blogs", "technology articles",
    "trending blogs", "trending articles", "expert insights", "thought leadership",
    "blogging", "blogging platform", "blog collections", "content hub", "knowledge base",
    "online articles", "digital articles", "company write-ups", "company insights"]
blog_reply = [
    "Here are some of our latest blogs:",
    "1. How to use MVP development to mitigate risk?: https://www.iotric.com/mvp-development-to-mitigate-risk/",
    "2. Minimum Viable Product (MVP) vs Minimum Marketable Product (MMP): https://www.iotric.com/mvp-vs-mmp/",
    "3. What is Fractional Ownership in Real Estate Investment with Blockchain?: https://www.iotric.com/what-is-fractional-ownership-in-real-estate-investment-with-blockchain/",
    "Visit our blog page https://www.iotric.com/blog/ for more articles!"
]


def get_greeting_reply(user_input):
    user_input = user_input.lower()
    if user_input in greetings:
        return reply
    
    if any(keyword in user_input for keyword in blog_keywords):
        return blog_reply

    return None


# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=1,
    return_messages=True
)
# retrieval qa chain


