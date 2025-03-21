from langchain.prompts import PromptTemplate

# Define a dynamic prompt template for financial analysis tasks
prompt_template = PromptTemplate(
    input_variables=["task", "context"],
    template="""
You are a financial analyst specializing in FAANG companies.
Task: {task}
Context: {context}
Provide a detailed analysis with actionable insights.
"""
)

# Example usage:
if __name__ == "__main__":
    task = "Summarize key risk factors."
    context = "Apple's 10-K filing mentions supplier dependency as a major risk."
    
    prompt = prompt_template.format(task=task, context=context)
    print(prompt)