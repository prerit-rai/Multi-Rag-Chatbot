import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def _create_sample_data(self):
        return {
            "wearable": [
                {"date": "2025-05-15", "steps": 8472, "sleep_quality": 91},
                {"date": "2025-05-19", "steps": 3215, "sleep_quality": 77}
            ],
            "chat_history": [
                {"user": "What's my sleep pattern?", "bot": "Average sleep quality: 85%"},
                {"user": "What's my yesterday activity?", "bot": "3215 steps taken"}
            ],
            "user_profile": {
                "name": "Prerit",
                "age": 21,
                "preferences": {"exercise": "morning"}
            },
            "location": [
                {"date": "2025-05-15", "places": ["College", "Gym"]}
            ],
            "custom": {
                "goals": {"target_steps": 10000},
                "health_metrics": {"BMI": 21.8}
            }
        }

    def load_documents(self):
        collections = self._create_sample_data()
        docs = []
        
        for collection_name, data in collections.items():
            if isinstance(data, list):
                for item in data:
                    text = json.dumps(item)
                    docs.extend(self._create_doc(text, collection_name))
            else:
                text = json.dumps(data)
                docs.extend(self._create_doc(text, collection_name))
                
        return docs

    def _create_doc(self, text, collection_name):
        return self.text_splitter.create_documents(
            [text],
            [{"collection": collection_name}]
        )
