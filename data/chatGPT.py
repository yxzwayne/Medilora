
import os
import sys
import pandas as pd

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
#   loader = TextLoader("data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-4"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)
data_path = 'patient_data.csv'
df = pd.read_csv(data_path, usecols=['notes'])
df['row_number'] = df.index

chat_history = []

def generate_qa_pair(data):
    idx = data['row_number']
    note = data['notes']
    query = f"Read this medical note: {note}. Generate 1 question and answer pair. Make sure the question-answer pair is not specific and is useful for patients or medical professionals. It can be about the particular disease, diagnosis, treatment, rest, recovery, key information, summary, symptoms, procedures, tests, etc., mentioned in the note. The Q&A should be of the form Question: [Q here] Answer: [A here]. I want to use these Q&As to fine-tune a medical LLM. Do not use specific information from this particular patient's case; only keep general information. Also limit the Answers to 100 words and Questions to 50 words. PLEASE DO NOT ADD DATA TO THE ANSWER ON YOUR OWN (DO NOT HALLUCINATE, STICK TO THE MEDICAL NOTE GIVEN IN THE PROMPT)"
    
    result = chain({"question": query, "chat_history": chat_history})
    result['answer'] = result['answer']
    print(idx)
    question_answer_pair = {
        "Q&A": result['answer']
    }
    
    return pd.Series(question_answer_pair)

# Apply the function to each row of the DataFrame
output_df = df.apply(generate_qa_pair, axis=1)

# Save the results to a new CSV file
output_df.to_csv("qa_pairs.txt", index=False)

