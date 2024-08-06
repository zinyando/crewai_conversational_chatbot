#!/usr/bin/env python
from crewai_conversational_chatbot.crew import CrewaiConversationalChatbotCrew
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "chatbot_memory",
            "path": "./chroma_db",
        },
    },
}

memory = Memory.from_config(config)


def run():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! It was nice talking to you.")
            break

        # Add user input to memory
        memory.add(f"User: {user_input}", user_id="user")

        # Retrieve relevant information from vector store
        relevant_info = memory.search(query=user_input, limit=3)
        context = "\n".join(message["memory"] for message in relevant_info)

        inputs = {
            "user_message": f"{user_input}",
            "context": f"{context}",
        }

        response = CrewaiConversationalChatbotCrew().crew().kickoff(inputs=inputs)

        # Add chatbot response to memory
        memory.add(f"Assistant: {response}", user_id="assistant")
        print(f"Assistant: {response}")
