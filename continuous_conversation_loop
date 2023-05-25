from message_processing import MessageProcessor
from memory_updates import MemoryUpdates
from language_model_interactions import LanguageModelInteractions

class ContinuousConversationLoop:
    def __init__(self, model, tokenizer):
        # Initialize components
        self.message_processor = MessageProcessor(model, tokenizer)
        self.memory_updates = MemoryUpdates()
        self.language_model_interactions = LanguageModelInteractions(model, tokenizer)

    def process_message(self, sender, receiver, message):
        # Retrieve personal memory of the sender
        personal_memory = self.memory_updates.get_personal_memory(sender)

        # Process the message and generate a response
        response = self.message_processor.process_message(sender, receiver, message, personal_memory)

        # Update personal memory of the sender
        self.memory_updates.update_personal_memory(sender, "last_message", message)

        return response

    def update_memory(self, sender, receiver, message):
        # Update collective memory
        conversation = (sender, receiver, message)
        self.memory_updates.update_collective_memory(conversation)

    def run(self):
        # Continuous conversation loop
        while True:
            # Get input from user
            sender = input("Sender: ")
            receiver = input("Receiver: ")
            message = input("Message: ")

            # Process the message and generate a response
            response = self.process_message(sender, receiver, message)

            # Update memory with the conversation
            self.update_memory(sender, receiver, message)

            # Print the response
            print("Response:", response)
            print()

if __name__ == "__main__":
    # Example usage
    # Load the language model and tokenizer (replace with your specific model and tokenizer)
    model = ...  # Load your language model
    tokenizer = ...  # Load your tokenizer

    # Create a continuous conversation loop instance
    conversation_loop = ContinuousConversationLoop(model, tokenizer)

    # Run the conversation loop
    conversation_loop.run()
