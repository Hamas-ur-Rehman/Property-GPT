import os
import time
import shutil
from tqdm import tqdm
from dotenv import load_dotenv
from utils.custom_logger import log
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
import threading
load_dotenv()


class ChromaService:
    def loader(self):
        try:
            start_time = time.time()
            # Initialize the embeddings function from OpenAI
            embeddings = OpenAIEmbeddings()

            # Initialize the Chroma vector store
            log.info("Initializing Chroma vector store")
            if os.path.exists("./chromadb"):
                log.info("Removing existing Chroma vector store")
                shutil.rmtree("./chromadb")
            vectorstore = Chroma(
                persist_directory="./chromadb",
                embedding_function=embeddings,
                collection_name="properties"
            )

            # Load the data from the CSV file
            log.info("Loading data from CSV file")
            loader = CSVLoader(file_path="./dataset/property.csv")
            data = loader.load()

            # Load the data into the Chroma vector store using multithreading
            log.info(f"Loading data into Chroma vector store. Please be patient, this may take a while... Total records: {len(data)}")
            threads = []
            for i in tqdm(data):
                thread = threading.Thread(target=Chroma.add_texts, args=(vectorstore, [i.page_content]))
                thread.start()
                threads.append(thread)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            log.info("Data loaded successfully")
            log.warning(f"Time taken to load data: {round(time.time() - start_time, 2)} seconds")
        except Exception as e:
            log.error(f"Error in loader: {e}", exc_info=True)


    def retriver(self, question, k=4):
        try:
            start_time = time.time()
            # Initialize the embeddings function from OpenAI
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(
                persist_directory="./chromadb",
                embedding_function=embeddings,
                collection_name="properties"
                )
            log.info("Retrieving documents from Chroma vector store")
            DOCS = []
            for i in vectorstore.similarity_search(question, k=k):
                DOCS.append(i.page_content)
            if len(DOCS) == 0:
                DOCS.append("Sorry, I couldn't find any relevant documents")
            log.warning(f"Time taken to retrieve documents: {round(time.time() - start_time,2)} seconds")
            return DOCS
        except Exception as e:
            log.error(f"Error in retriver: {e}", exc_info=True)
            return ["Sorry, I couldn't find any relevant documents"]
