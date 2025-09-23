import os
import httpx
import asyncio
import math
import re
from dotenv import load_dotenv  # Add this import
from mcp.server.fastmcp import FastMCP  # use FastMCP instead of Server
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY environment variable")

mcp = FastMCP("groq-agentic-mcp")

# Pydantic models for input validation
class GroqInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="The prompt to send to Groq API")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or just whitespace")
        return v.strip()

class WikipediaInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query for Wikipedia")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        # Remove potentially harmful characters
        cleaned = re.sub(r'[<>\"\'&]', '', v.strip())
        if len(cleaned) == 0:
            raise ValueError("Query contains only invalid characters")
        return cleaned

class CalculateInput(BaseModel):
    expression: str = Field(..., min_length=1, max_length=1000, description="Mathematical expression to evaluate")
    
    @field_validator('expression')
    @classmethod
    def validate_expression(cls, v):
        if not v.strip():
            raise ValueError("Expression cannot be empty or just whitespace")
        
        # Only allow safe characters for mathematical expressions
        allowed_chars = set('0123456789+-*/.() abcdefghijklmnopqrstuvwxyz_')
        expression = v.strip().lower()
        
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")
        
        # Check for potentially dangerous patterns
        dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file', 'input']
        if any(pattern in expression for pattern in dangerous_patterns):
            raise ValueError("Expression contains potentially dangerous operations")
        
        return expression

class AgentInput(BaseModel):
    task: str = Field(..., min_length=1, max_length=5000, description="Task description for the agent")
    
    @field_validator('task')
    @classmethod
    def validate_task(cls, v):
        if not v.strip():
            raise ValueError("Task cannot be empty or just whitespace")
        return v.strip()

# Tool 1: ask_groq
@mcp.tool()
async def ask_groq(prompt: str) -> str:
    # Validate input
    try:
        validated_input = GroqInput(prompt=prompt)
    except Exception as e:
        return f"Input validation error: {str(e)}"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": validated_input.prompt}],
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"

# Tool 2: search_wikipedia
@mcp.tool()
async def search_wikipedia(query: str) -> str:
    # Validate input
    try:
        validated_input = WikipediaInput(query=query)
    except Exception as e:
        return f"Input validation error: {str(e)}"
    
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{validated_input.query}"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return f"No result found for {validated_input.query}"
            data = r.json()
        return data.get("extract", "No summary available")
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

# Tool 3: calculate
@mcp.tool()
async def calculate(expression: str) -> str:
    # Validate input
    try:
        validated_input = CalculateInput(expression=expression)
    except Exception as e:
        return f"Input validation error: {str(e)}"
    
    try:
        safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(validated_input.expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error in calculation: {e}"

# Tool 4: agent
@mcp.tool()
async def agent(task: str) -> str:
    # Validate input
    try:
        validated_input = AgentInput(task=task)
    except Exception as e:
        return f"Input validation error: {str(e)}"
    
    decision_prompt = f"""
    You are an agent. A user asked: "{validated_input.task}".
    
    Analyze the request and choose the best approach:
    1. If it's asking for a mathematical calculation, respond with: CALCULATE:<math_expression>
    2. If it's asking for factual information or knowledge lookup, respond with: WIKIPEDIA:<search_term>
    3. For anything else (reasoning, explanation, general questions), respond with: GROQ:<original_question>
    
    Examples:
    - "What is 1+1?" -> CALCULATE:1+1
    - "What is 2*3+5?" -> CALCULATE:2*3+5
    - "Tell me about Einstein" -> WIKIPEDIA:Einstein
    - "Who is the president of France?" -> WIKIPEDIA:President of France
    - "Explain quantum physics" -> GROQ:Explain quantum physics
    
    Respond with the appropriate format for: "{validated_input.task}"
    """
    
    try:
        decision = await ask_groq(decision_prompt)
        decision = decision.strip()
        
        if decision.startswith("CALCULATE:"):
            expression = decision[10:]  # Remove "CALCULATE:" prefix
            return await calculate(expression)
        elif decision.startswith("WIKIPEDIA:"):
            search_term = decision[10:]  # Remove "WIKIPEDIA:" prefix
            return await search_wikipedia(search_term)
        elif decision.startswith("GROQ:"):
            question = decision[5:]  # Remove "GROQ:" prefix
            return await ask_groq(question)
        else:
            # Fallback - if the format isn't recognized, just pass to ask_groq
            return await ask_groq(validated_input.task)
    except Exception as e:
        return f"Error processing agent request: {str(e)}"

if __name__ == "__main__":
    # Run server using FastMCP's run method
    mcp.run(transport="stdio")  # or other transport if you want like HTTP