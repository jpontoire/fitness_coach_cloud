from agent.nodes import load_resources
from agent.graph import build_graph

def main():
    app = build_graph()

    test_questions = [
        ("Structured", "How do I do a dumbbell bench press?"),
        ("Program", "Give me a push day workout with dumbbells"),
        ("Conversational", "How much protein should I eat to build muscle?"),
        ("Off-topic", "What's the capital of Italy?"),
    ]

    for label, question in test_questions:
        print(f"=== {label} ===")
        result = app.invoke({"question": question})
        print(f"Intent: {result.get('intent')}")
        print(f"Answer: {result.get('answer')}")
        print("---")

if __name__ == "__main__":
    main()
