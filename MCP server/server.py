
import os
import httpx
import asyncio
import math
import re
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator
from urllib.parse import quote

# Configure logging with structured format
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress non-essential logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY environment variable")

mcp = FastMCP("groq-agentic-mcp")

# Pydantic models for input validation
class GroqInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="Prompt for Groq API")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or just whitespace")
        return v.strip()

class WeatherInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Location query for weather API")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
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
        allowed_chars = set('0123456789+-*/.() abcdefghijklmnopqrstuvwxyz_')
        expression = v.strip().lower()
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")
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

class ExecutionTrace(BaseModel):
    subtask_id: str
    tool_used: str
    input_data: str
    output_data: str
    execution_time: float
    status: str  # "success" or "error"
    
    def trim_output(self, max_length: int = 200) -> str:
        if len(self.output_data) <= max_length:
            return self.output_data
        return self.output_data[:max_length] + "... [truncated]"

async def execute_tool_with_trace(tool_name: str, input_data: str, subtask_id: str) -> ExecutionTrace:
    import time
    start_time = time.time()
    
    logger.info(f"Executing {tool_name} for subtask {subtask_id}")
    logger.info(f"Input: {input_data}")
    
    try:
        if tool_name == "CALCULATE":
            result = await calculate(input_data, subtask_id)
        elif tool_name == "WEATHER":
            result = await get_weather(input_data, subtask_id)
        elif tool_name == "GROQ":
            result = await ask_groq(input_data)
        else:
            result = f"Unknown tool: {tool_name}"
            
        execution_time = time.time() - start_time
        logger.info(f"Output: {result[:200]}{'...' if len(result) > 200 else ''}")
        logger.info(f"Status: success, Time: {execution_time:.2f}s")
        return ExecutionTrace(
            subtask_id=subtask_id,
            tool_used=tool_name,
            input_data=input_data,
            output_data=result,
            execution_time=execution_time,
            status="success"
        )
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Output: Error: {repr(str(e))}")
        logger.error(f"Status: error, Time: {execution_time:.2f}s")
        return ExecutionTrace(
            subtask_id=subtask_id,
            tool_used=tool_name,
            input_data=input_data,
            output_data=f"Error: {str(e)}",
            execution_time=execution_time,
            status="error"
        )

@mcp.tool()
async def ask_groq(prompt: str) -> str:
    try:
        validated_input = GroqInput(prompt=prompt)
    except Exception as e:
        logger.error(f"Input validation error in ask_groq: {repr(str(e))}")
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
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        logger.error(f"Error calling Groq API: {repr(str(e))}")
        return f"Error calling Groq API: {str(e)}"

@mcp.tool()
async def get_weather(location: str, subtask_id: str) -> str:
    try:
        validated_input = WeatherInput(query=location)
    except Exception as e:
        logger.error(f"Input validation error in get_weather: {repr(str(e))}")
        return f"Input validation error: {str(e)}"
    
    current_query = validated_input.query
    attempted_queries = set([current_query])
    max_retries = 3
    
    for attempt in range(max_retries):
        params = {"name": quote(current_query), "count": 1, "format": "json", "language": "en"}
        logger.debug(f"Attempt {attempt + 1}/{max_retries} geocoding with query: '{current_query}'")
        
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(geocoding_url, params=params)
                response.raise_for_status()
                data = response.json()
            if not data.get("results"):
                raise ValueError(f"No location found for '{current_query}'")
            latitude = data["results"][0]["latitude"]
            longitude = data["results"][0]["longitude"]
            city = data["results"][0]["name"]
            country = data["results"][0].get("country", "Unknown")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} geocoding failed for '{current_query}': {repr(str(e))}")
            if attempt < max_retries - 1:
                refine_prompt = f"""
                The weather query '{current_query}' failed with error: '{str(e)}'.
                Previous attempts: {', '.join(attempted_queries)}.
                Suggest a specific location query for the weather API, preferably a city name or city,country.
                Avoid previously tried queries.
                Output the refined query as plain text.
                """
                try:
                    current_query = await ask_groq(refine_prompt)
                    if current_query in attempted_queries:
                        logger.warning(f"Groq suggested duplicate query '{current_query}', skipping")
                        current_query = validated_input.query
                    else:
                        attempted_queries.add(current_query)
                        logger.debug(f"Refined query for attempt {attempt + 2}: {current_query}")
                    continue
                except Exception as refine_e:
                    logger.error(f"Error refining query for '{current_query}': {repr(str(refine_e))}")
                    continue
            else:
                logger.error(f"Max retries reached for '{current_query}'")
                return f"Error: Failed to find location '{validated_input.query}' after {max_retries} attempts: {str(e)}"
        
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "timezone": "auto"
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(weather_url, params=params)
                response.raise_for_status()
                data = response.json()
            temperature = data["current"]["temperature_2m"]
            weather_code = data["current"]["weather_code"]
            
            weather_codes = {
                0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "fog", 48: "depositing rime fog",
                51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
                61: "light rain", 63: "moderate rain", 65: "heavy rain",
                71: "light snow", 73: "moderate snow", 75: "heavy snow",
                80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers"
            }
            weather_desc = weather_codes.get(weather_code, "unknown")
            result = f"Current weather in {city}, {country}: {weather_desc}, temperature: {temperature}°C"
            return result
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} weather fetch failed for '{current_query}': {repr(str(e))}")
            if attempt < max_retries - 1:
                refine_prompt = f"""
                The weather query '{current_query}' failed with error: '{str(e)}'.
                Previous attempts: {', '.join(attempted_queries)}.
                Suggest a specific location query for the weather API, preferably a city name or city,country.
                Avoid previously tried queries.
                Output the refined query as plain text.
                """
                try:
                    current_query = await ask_groq(refine_prompt)
                    if current_query in attempted_queries:
                        logger.warning(f"Groq suggested duplicate query '{current_query}', skipping")
                        current_query = validated_input.query
                    else:
                        attempted_queries.add(current_query)
                        logger.debug(f"Refined query for attempt {attempt + 2}: {current_query}")
                    continue
                except Exception as refine_e:
                    logger.error(f"Error refining query for '{current_query}': {repr(str(refine_e))}")
                    continue
            else:
                logger.error(f"Max retries reached for '{current_query}'")
                return f"Error: Failed to fetch weather for '{validated_input.query}' after {max_retries} attempts: {str(e)}"

@mcp.tool()
async def calculate(expression: str, subtask_id: str) -> str:
    max_retries = 3
    current_expression = expression
    attempted_expressions = set([current_expression])
    
    for attempt in range(max_retries):
        try:
            validated_input = CalculateInput(expression=current_expression)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} validation error in calculate for '{current_expression}': {repr(str(e))}")
            if attempt < max_retries - 1:
                refine_prompt = f"""
                The calculation expression '{current_expression}' failed with error: '{str(e)}'.
                Previous attempts: {', '.join(attempted_expressions)}.
                Suggest a corrected mathematical expression using only numbers, operators (+, -, *, /), and valid mathematical functions.
                Output the refined expression as plain text.
                """
                try:
                    current_expression = await ask_groq(refine_prompt)
                    if current_expression in attempted_expressions:
                        logger.warning(f"Groq suggested duplicate expression '{current_expression}', skipping")
                        current_expression = expression
                    else:
                        attempted_expressions.add(current_expression)
                        logger.debug(f"Refined calculate input for attempt {attempt + 2}: {current_expression}")
                    continue
                except Exception as refine_e:
                    logger.error(f"Error refining expression for '{current_expression}': {repr(str(refine_e))}")
                    continue
            else:
                logger.error(f"Max retries reached for '{current_expression}'")
                return f"Error: Failed to calculate expression '{expression}' after {max_retries} attempts: {str(e)}"
        
        try:
            safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            result = eval(validated_input.expression, {"__builtins__": {}}, safe_dict)
            return str(result)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} evaluation error in calculate for '{current_expression}': {repr(str(e))}")
            if attempt < max_retries - 1:
                refine_prompt = f"""
                The calculation expression '{current_expression}' failed with error: '{str(e)}'.
                Previous attempts: {', '.join(attempted_expressions)}.
                Suggest a corrected mathematical expression using only numbers, operators (+, -, *, /), and valid mathematical functions.
                Output the refined expression as plain text.
                """
                try:
                    current_expression = await ask_groq(refine_prompt)
                    if current_expression in attempted_expressions:
                        logger.warning(f"Groq suggested duplicate expression '{current_expression}', skipping")
                        current_expression = expression
                    else:
                        attempted_expressions.add(current_expression)
                        logger.debug(f"Refined calculate input for attempt {attempt + 2}: {current_expression}")
                    continue
                except Exception as refine_e:
                    logger.error(f"Error refining expression for '{current_expression}': {repr(str(refine_e))}")
                    continue
            else:
                logger.error(f"Max retries reached for '{current_expression}'")
                return f"Error: Failed to calculate expression '{expression}' after {max_retries} attempts: {str(e)}"

@mcp.tool()
async def agent(task: str) -> str:
    try:
        validated_input = AgentInput(task=task)
    except Exception as e:
        logger.error(f"Input validation error in agent: {repr(str(e))}")
        return f"Input validation error: {str(e)}"
    
    # Initialize state
    completed_tasks = {}  # Store subtask_id: output
    execution_traces = []
    task_counter = 0  # For generating subtask IDs
    
    while True:
        task_counter += 1
        subtask_id = f"task_{task_counter}"
        
        # Classify the next tool(s)
        classify_prompt = f"""
        You are a task classification agent using chain-of-thought reasoning.

        Original task: "{validated_input.task}"
        Previous results:
        {json.dumps(completed_tasks, indent=2) if completed_tasks else "None"}

        Follow these steps:
        1. Analyze the original task and previous results to identify all required subtasks.
        2. Determine if the subtasks are independent (can be executed concurrently) or dependent (must be executed sequentially).
        3. Select the appropriate tool(s) for the next subtask(s): CALCULATE, WEATHER, or GROQ.
           - CALCULATE: For numerical computations. Input must be a valid mathematical expression (numbers, operators, math functions).
           - WEATHER: For weather queries. Input must be a specific location (e.g., city or city,country).
           - GROQ: For general reasoning, text processing, or final answers.
        4. If all subtasks are complete based on previous results, output {{"tool": "DONE", "input": "Final answer"}}.
        5. If subtasks are independent and none depend on uncompleted results, output a JSON array of all tools and their inputs to execute them concurrently.
        6. If subtasks are dependent or only one subtask is ready, output a single JSON object for the next tool.
        7. Provide exact inputs for the selected tool(s), using previous results if needed. Ensure inputs are valid for the tool (e.g., no placeholders, correct format).
        8. If conversions (e.g., units) are needed, include them in the input.
        9. If inputs require data from previous results, extract and format appropriately.

        Output EXACTLY in one of the following formats, wrapped in <json> tags, with no extra text, comments, or explanations:
        - For a single tool (sequential execution):
          <json>
          {{"tool": "tool_name", "input": "input_string"}}
          </json>
        - For multiple independent tools (concurrent execution):
          <json>
          [{{"tool": "tool_name1", "input": "input_string1"}}, {{"tool": "tool_name2", "input": "input_string2"}}]
          </json>
        - For completion:
          <json>
          {{"tool": "DONE", "input": "Final answer"}}
          </json>
        Ensure the output is valid JSON with proper escaping and no additional content outside the <json> tags.
        """
        
        try:
            classify_response = await ask_groq(classify_prompt)
            logger.debug(f"Raw classify_response for {subtask_id}: {repr(classify_response)}")
            
            # Extract JSON
            json_start = classify_response.find('<json>')
            json_end = classify_response.rfind('</json>')
            
            if json_start != -1 and json_end != -1:
                json_start += len('<json>')
                json_str = classify_response[json_start:json_end].strip()
                logger.debug(f"Extracted JSON string for {subtask_id}: {repr(json_str)}")
            else:
                logger.warning(f"No <json> tags found in classify_response for {subtask_id}, attempting to parse from ```json block")
                json_match = re.search(r'```json\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```', classify_response, re.MULTILINE)
                if json_match:
                    json_str = json_match.group(1).strip()
                    logger.debug(f"Extracted JSON string from ```json block for {subtask_id}: {repr(json_str)}")
                else:
                    logger.error(f"No valid JSON block found in classification for {subtask_id}")
                    # Fallback to GROQ for the original task
                    logger.warning(f"Falling back to GROQ for task: {validated_input.task}")
                    final_answer = await ask_groq(validated_input.task)
                    response_parts = []
                    response_parts.append("🤖 **Agent Execution Trace**\n")
                    for trace in execution_traces:
                        status_emoji = "✅" if trace.status == "success" else "❌"
                        response_parts.append(f"{status_emoji} **{trace.tool_used}** (`{trace.subtask_id}`)")
                        response_parts.append(f"   📝 Input: {trace.input_data}")
                        response_parts.append(f"   📤 Output: {trace.trim_output()}")
                        response_parts.append(f"   ⏱️ Time: {trace.execution_time:.2f}s")
                        response_parts.append("")
                    response_parts.append("🎯 **Final Answer:**")
                    response_parts.append(final_answer)
                    return "\n".join(response_parts)
            
            # Parse JSON and handle potential errors
            try:
                classify_data = json.loads(json_str)
                logger.debug(f"Parsed classify_data for {subtask_id}: {classify_data}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse failed for {subtask_id}: {repr(str(e))}")
                # Fallback to GROQ for the original task
                logger.warning(f"Falling back to GROQ for task: {validated_input.task}")
                final_answer = await ask_groq(validated_input.task)
                response_parts = []
                response_parts.append("🤖 **Agent Execution Trace**\n")
                for trace in execution_traces:
                    status_emoji = "✅" if trace.status == "success" else "❌"
                    response_parts.append(f"{status_emoji} **{trace.tool_used}** (`{trace.subtask_id}`)")
                    response_parts.append(f"   📝 Input: {trace.input_data}")
                    response_parts.append(f"   📤 Output: {trace.trim_output()}")
                    response_parts.append(f"   ⏱️ Time: {trace.execution_time:.2f}s")
                    response_parts.append("")
                response_parts.append("🎯 **Final Answer:**")
                response_parts.append(final_answer)
                return "\n".join(response_parts)
                
            # Handle both single tool (dict) and multiple tools (list)
            if isinstance(classify_data, dict):
                if "tool" not in classify_data or "input" not in classify_data:
                    raise ValueError("Missing tool or input in classification response")
                tasks = [classify_data]
            elif isinstance(classify_data, list):
                if not classify_data:
                    raise ValueError("Empty task list in classification response")
                for task in classify_data:
                    if not isinstance(task, dict) or "tool" not in task or "input" not in task:
                        raise ValueError("Invalid task format in classification response")
                tasks = classify_data
            else:
                raise ValueError("Classification response must be a dict or list")
            
            # Check for completion
            if len(tasks) == 1 and tasks[0]["tool"] == "DONE":
                # Synthesize final answer
                synthesis_prompt = f"""
                Original task: "{validated_input.task}"
                
                Results from previous steps:
                {json.dumps(completed_tasks, indent=2) if completed_tasks else "None"}
                
                Provide a comprehensive answer to the original task using the results.
                """
                logger.debug(f"Synthesis prompt: {synthesis_prompt}")
                final_answer = await ask_groq(synthesis_prompt)
                
                response_parts = []
                response_parts.append("🤖 **Agent Execution Trace**\n")
                for trace in execution_traces:
                    status_emoji = "✅" if trace.status == "success" else "❌"
                    response_parts.append(f"{status_emoji} **{trace.tool_used}** (`{trace.subtask_id}`)")
                    response_parts.append(f"   📝 Input: {trace.input_data}")
                    response_parts.append(f"   📤 Output: {trace.trim_output()}")
                    response_parts.append(f"   ⏱️ Time: {trace.execution_time:.2f}s")
                    response_parts.append("")
                response_parts.append("🎯 **Final Answer:**")
                response_parts.append(final_answer)
                return "\n".join(response_parts)
            
            # Execute tasks (concurrent for list, sequential for single)
            if len(tasks) > 1:
                # Concurrent execution for independent tasks
                logger.info(f"Executing {len(tasks)} independent tasks concurrently for {subtask_id}")
                traces = await asyncio.gather(
                    *[execute_tool_with_trace(task["tool"], task["input"], f"{subtask_id}_{i+1}")
                      for i, task in enumerate(tasks)],
                    return_exceptions=True
                )
                for trace in traces:
                    if isinstance(trace, ExecutionTrace):
                        execution_traces.append(trace)
                        completed_tasks[trace.subtask_id] = trace.output_data
                        if trace.status == "error":
                            response_parts = []
                            response_parts.append("🤖 **Agent Execution Trace**\n")
                            for t in execution_traces:
                                status_emoji = "✅" if t.status == "success" else "❌"
                                response_parts.append(f"{status_emoji} **{t.tool_used}** (`{t.subtask_id}`)")
                                response_parts.append(f"   📝 Input: {t.input_data}")
                                response_parts.append(f"   📤 Output: {t.trim_output()}")
                                response_parts.append(f"   ⏱️ Time: {t.execution_time:.2f}s")
                                response_parts.append("")
                            response_parts.append("🎯 **Answer:**")
                            response_parts.append(trace.output_data)
                            return "\n".join(response_parts)
            else:
                # Sequential execution for a single task
                trace = await execute_tool_with_trace(tasks[0]["tool"], tasks[0]["input"], subtask_id)
                execution_traces.append(trace)
                completed_tasks[subtask_id] = trace.output_data
                if trace.status == "error":
                    response_parts = []
                    response_parts.append("🤖 **Agent Execution Trace**\n")
                    for t in execution_traces:
                        status_emoji = "✅" if t.status == "success" else "❌"
                        response_parts.append(f"{status_emoji} **{t.tool_used}** (`{t.subtask_id}`)")
                        response_parts.append(f"   📝 Input: {t.input_data}")
                        response_parts.append(f"   📤 Output: {t.trim_output()}")
                        response_parts.append(f"   ⏱️ Time: {t.execution_time:.2f}s")
                        response_parts.append("")
                    response_parts.append("🎯 **Answer:**")
                    response_parts.append(trace.output_data)
                    return "\n".join(response_parts)
        
        except Exception as e:
            logger.error(f"Error in agent processing for {subtask_id}: {repr(str(e))}")
            return f"Error in agent processing for {subtask_id}: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
