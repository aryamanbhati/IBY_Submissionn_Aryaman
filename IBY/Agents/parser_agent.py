from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import json
import re

def parse_resume(resume_data):
    template = """
    You are an AI bot designed to act as a professional resume parser. Your job is to extract the following information from the resume and provide it in a single JSON object.

    Strict Instructions:
    - Respond ONLY with the JSON object. Do not include any other text, explanations, or code block delimiters.
    - If a field is not found, use a null value for that key.

    JSON Schema to follow:
    {
      "full_name": "string or null",
      "email_id": "string or null",
      "github_portfolio": "string or null",
      "linkedin_id": "string or null",
      "employment_details": [
        {
          "company_name": "string",
          "job_title": "string",
          "start_date": "string",
          "end_date": "string",
          "responsibilities": "string"
        }
      ],
      "technical_skills": [
        "string"
      ],
      "soft_skills": [
        "string"
      ]
    }
    Resume:
    {resume_text}
    """
    model_name = "llama3:latest"
    base_url = "http://localhost:11434"
    llm = OllamaLLM(model=model_name, base_url=base_url)

    prompt = PromptTemplate.from_template(template)
    final_prompt = prompt.format(resume_text=resume_data)
    try:
        response = llm(final_prompt)
    except Exception as e:
        return {"error": f"LLM call failed: {e}"}

    # Clean the response to ensure it's valid JSON
    cleaned_response = response.strip()

    # Remove any code block formatting or extra text
    match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
    if match:
        json_string = match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse JSON: {e}",
                "raw_response": cleaned_response
            }
    else:
        return {
            "error": "No JSON object found in the LLM response.",
            "raw_response": cleaned_response
        }

# Example usage (assuming you have a resume string)
# resume_text = "..."
# parsed_data = parse_resume(resume_text)
# print(parsed_data)