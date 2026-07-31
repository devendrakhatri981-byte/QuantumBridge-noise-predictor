import sys

# Get user's name
name = input("What's your name? ")
print(f"Hello, {name}!\n")

# Continuous chatbot
print("Chatbot started! Type 'exit' to quit.")
print("-" * 40)

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            sys.exit(0)
        
        if not user_input:
            continue
            
        print(f"Chatbot: You said '{user_input}'. That's interesting!")
        
    except EOFError:
        print("\nChatbot: Goodbye!")
        break
    except KeyboardInterrupt:
        print("\n\nChatbot: Goodbye!")
        break