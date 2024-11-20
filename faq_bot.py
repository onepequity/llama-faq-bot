import os
from transformers import pipeline

# Fetch token from environment variable (optional if using a public model)
hf_token = os.getenv("HF_TOKEN")
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", token=hf_token if hf_token else None)

context = """
LLaMA is an open-source language model by Meta, designed for research and business applications.
It excels in text generation, summarization, and question-answering tasks. LLaMA models are available in various sizes, including 7B, 13B, and 70B parameters. They are optimized for efficiency and performance, often outperforming other open-source models in benchmarks. LLaMA can be used to protect proprietary information because it is open-source, allowing companies to host it on their own infrastructure, ensuring data stays within their control and is not shared with third-party providers, unlike some competitive LLMs like ChatGPT, which may use data for training. This makes LLaMA a secure choice for businesses handling sensitive information.
"""

def answer_question(question):
    try:
        if not question.strip():  # Check for empty input etc
            return "Please ask a question!"
        result = qa_model(question=question, context=context)
        # Add a confidence threshold to detect irrelevant answers
        if result['score'] < 0.1:  # Adjust threshold as needed
            return "Sorry, I can’t answer that based on the information I have."
        return result['answer']
    except Exception as e:
        return f"Sorry, I couldn’t find an answer to your question. Error: {e}"

if __name__ == "__main__":
    print("Device set to use CPU")  # Since you mentioned CPU usage
    while True:
        question = input("Ask a question about LLaMA (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = answer_question(question)
        print(f"Answer: {answer}")