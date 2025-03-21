from kafka import KafkaProducer
import os

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI API key from environment variables
kafka_broker_url = os.getenv("KAFKA_BROKER_URL")

producer = KafkaProducer(bootstrap_servers=kafka_broker_url)

def stream_query(query):
    producer.send('financial_queries', value=query.encode('utf-8'))

# Example usage:
if __name__ == "__main__":
    stream_query("What are Apple's key risks?")