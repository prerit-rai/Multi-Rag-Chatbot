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
                {"date": "2023-12-01", "steps": 8432, "sleep_quality": 85},
                {"date": "2023-12-02", "steps": 9215, "sleep_quality": 88}
            ],
            "chat_history": [
                {"user": "What's my sleep pattern?", "bot": "Average sleep quality: 86%"},
                {"user": "Yesterday's activity?", "bot": "9215 steps taken"}
            ],
            "user_profile": {
                "name": "Alex",
                "age": 32,
                "preferences": {"exercise": "evening"}
            },
            "location": [
                {"date": "2023-12-01", "places": ["Office", "Gym"]}
            ],
            "custom": {
                "goals": {"target_steps": 10000},
                "health_metrics": {"BMI": 24.3}
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