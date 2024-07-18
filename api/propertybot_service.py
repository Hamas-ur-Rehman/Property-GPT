from services.chroma_service import retriver
from services.openai_service import askai
from services.mongodb_service import *
from utils.custom_logger import log
from prompts.property_prompt import PROPERTY_PROMPT
import datetime

class PropertyBotService:
    def __init__(self):
        chromadb_service = ChromaService()
        openai_service = OpenAIService()
        mongodb_service = MongoDBService()

    def ask(self,userid:str,question:str):

        # Insert user question into MongoDB database
        try:
            mongodb_service.insert_chat(
                userid=userid,
                role="user",
                msg=question,
                created_at=datetime.datetime.now()
            )
        except Exception as e:
            log.error(f"Error in insert_chat: {e}", exc_info=True)


        # Retrieve documents from the Chroma VectorDatabase
        docs = chromadb_service.retriver(question)
        
        # Prepare Dynamic Prompt with the Data from VectorDatabase
        DOC_PROMPT = 'This is from where you can possibly get information about the user question'
        for doc in docs:
            DOC_PROMPT += "\n" + doc
        messages = [
            {
                "role": "system",
                "content": PROPERTY_PROMPT
            },
            {
                "role": "system",
                "content": DOC_PROMPT
            },
        ]

        # Fetch previous chats from the MongoDB database
        try:
            chats = mongodb_service.fetch_chats(userid)
            for chat in chats:
                    # Append the previous chats to the messages list just like OpenAI format
                    messages.append(
                    {
                        "role":chat['role'],
                        "content":chat["msg"]
                    }
                    )
        except Exception as e:
            log.error(f"Error in fetch_chats: {e}", exc_info=True)
        
        # Append the user question to the messages list just like OpenAI format
        messages.append({"role":"user","content":question})

        # Get response from OpenAI
        response = openai_service.askai(messages)

        # Insert the response into MongoDB database
        try:
            mongodb_service.insert_chat(
                userid=userid,
                role="assistant",
                msg=response,
                created_at=datetime.datetime.now()
            )
        except Exception as e:
            log.error(f"Error in insert_chat: {e}", exc_info=True)
        return response
