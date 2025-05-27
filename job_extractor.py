"""
TODO: Create a job extractor using structured output

Instructions:
1. Import necessary modules (dotenv, ChatGoogleGenerativeAI, ChatPromptTemplate)
2. Import your JobPosting model
3. Create extract_job_posting function that:
   - Initializes the LLM
   - Creates structured LLM with with_structured_output()
   - Creates a prompt template
   - Builds and invokes the chain

See README.md for detailed step-by-step instructions.
"""

# TODO: Add your imports here
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from job_models import JobPosting

# TODO: Add load_dotenv() call here
load_dotenv()

# Check if API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# TODO: Create your extract_job_posting function here
def extract_job_posting(text: str) -> JobPosting:
    """Extract job posting information from text using structured output"""
    
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17", temperature=0
        )
        # Create structured LLM
        structured_llm = llm.with_structured_output(JobPosting)
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(
            """
        Extract job posting information from the following text.

        Focus on identifying:
        - Job title
        - Company name
        - Location
        - Employment type (full-time, part-time, contract, etc.)
        - Brief summary of the role
        - Key requirements
        - Salary range if mentioned

        Text to analyze:
        {text}
        """
        )
        # Build and invoke chain
        chain = prompt | structured_llm
        result = chain.invoke({"text": text})

        return result
    except Exception as e:
        print(f"Error extracting job posting: {str(e)}")
        raise

# Test the function
if __name__ == "__main__":
    test_text = "Software Engineer at TechCorp in SF. Full-time, $100k."
    result = extract_job_posting(test_text)
    print(result.model_dump())