import os
import asyncio
import time
import json
import logging
import re
import grpc
import grpc.aio
from typing import Dict, Any, List
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator
from tools_pb2 import StringRequest, StringResponse
from tools_pb2_grpc import GroqServiceStub, WeatherServiceStub, CalculateServiceStub

# Logging setup (unchanged)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logging.getLogger("mcp").setLevel(logging.WARNING)

load_dotenv()

mcp = FastMCP("groq-agentic-mcp")

# Configurable hosts for scaling
GROQ_SERVICE_HOST = "localhost:50051"
WEATHER_SERVICE_HOST = "localhost:50052"
CALCULATE_SERVICE_HOST = "localhost:50053"

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
    start_time = time.time()
    logger.info(f"Executing {tool_name} for subtask {subtask_id}")
    logger.info(f"Input: {input_data}")
    
    try:
        if tool_name == "CALCULATE":
            async with grpc.aio.insecure_channel(CALCULATE_SERVICE_HOST) as channel:
                stub = CalculateServiceStub(channel)
                response = await stub.Calculate(StringRequest(value=input_data))  # Remove explicit type for debugging
            logger.debug(f"Calculate response type: {type(response)}, content: {response}")
            if hasattr(response, 'error') and response.error:  # Safely check for error field
                raise ValueError(response.error)
            result = response.result if hasattr(response, 'result') else str(response)  # Fallback to string if proto fails
        elif tool_name == "WEATHER":
            async with grpc.aio.insecure_channel(WEATHER_SERVICE_HOST) as channel:
                stub = WeatherServiceStub(channel)
                response = await stub.GetWeather(StringRequest(value=input_data))  # Remove explicit type for debugging
            logger.debug(f"Weather response type: {type(response)}, content: {response}")
            if hasattr(response, 'error') and response.error:  # Safely check for error field
                raise ValueError(response.error)
            result = response.result if hasattr(response, 'result') else str(response)  # Fallback to string if proto fails
        elif tool_name == "GROQ":
            async with grpc.aio.insecure_channel(GROQ_SERVICE_HOST) as channel:
                stub = GroqServiceStub(channel)
                response = await stub.AskGroq(StringRequest(value=input_data))  # Remove explicit type for debugging
            logger.debug(f"Groq response type: {type(response)}, content: {response}")
            if hasattr(response, 'error') and response.error:  # Safely check for error field
                raise ValueError(response.error)
            result = response.result if hasattr(response, 'result') else str(response)  # Fallback to string if proto fails
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        execution_time = time.time() - start_time
        logger.info(f"Output: {result[:200]}{'...' if len(result) > 200 else ''}")
        logger.info(f"Status: success, Time: {execution_time:.2f}s")
        return ExecutionTrace(subtask_id=subtask_id, tool_used=tool_name, input_data=input_data, output_data=result, execution_time=execution_time, status="success")
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Output: Error: {str(e)}")
        logger.error(f"Status: error, Time: {execution_time:.2f}s")
        return ExecutionTrace(subtask_id=subtask_id, tool_used=tool_name, input_data=input_data, output_data=f"Error: {str(e)}", execution_time=execution_time, status="error")

@mcp.tool()
async def agent(task: str) -> str:
    try:
        validated_input = AgentInput(task=task)
    except Exception as e:
        logger.error(f"Input validation error in agent: {str(e)}")
        return f"Input validation error: {str(e)}"
    
    # State (unchanged)
    completed_tasks = {}
    execution_traces = []
    task_counter = 0
    
    while True:
        task_counter += 1
        subtask_id = f"task_{task_counter}"
        
        # Classify using GroqService via gRPC (unchanged)
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
            # Call GroqService
            async with grpc.aio.insecure_channel(GROQ_SERVICE_HOST) as channel:
                stub = GroqServiceStub(channel)
                classify_response = await stub.AskGroq(StringRequest(value=classify_prompt))  # Remove explicit type for debugging
            logger.debug(f"Classify response type: {type(classify_response)}, content: {classify_response}")
            if hasattr(classify_response, 'error') and classify_response.error:  # Safely check for error field
                raise ValueError(classify_response.error)
            classify_response = classify_response.result if hasattr(classify_response, 'result') else str(classify_response)  # Fallback to string
            logger.debug(f"Raw classify_response for {subtask_id}: {repr(classify_response)}")
            
            # Extract and parse JSON (unchanged)
            json_start = classify_response.find('<json>')
            json_end = classify_response.rfind('</json>')
            if json_start != -1 and json_end != -1:
                json_start += len('<json>')
                json_str = classify_response[json_start:json_end].strip()
            else:
                json_match = re.search(r'```json\s*(\{[\s\S]*?\}|$$   [\s\S]*?   $$)\s*```', classify_response, re.MULTILINE)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    raise ValueError("No valid JSON in classification")
            
            classify_data = json.loads(json_str)
            
            if isinstance(classify_data, dict):
                tasks = [classify_data]
            elif isinstance(classify_data, list):
                tasks = classify_data
            else:
                raise ValueError("Invalid classification format")
            
            if len(tasks) == 1 and tasks[0]["tool"] == "DONE":
                # Synthesize using GroqService
                synthesis_prompt = f"""
                Original task: "{validated_input.task}"
                
                Results from previous steps:
                {json.dumps(completed_tasks, indent=2) if completed_tasks else "None"}
                
                Provide a comprehensive answer to the original task using the results.
                """
                async with grpc.aio.insecure_channel(GROQ_SERVICE_HOST) as channel:
                    stub = GroqServiceStub(channel)
                    response = await stub.AskGroq(StringRequest(value=synthesis_prompt))  # Remove explicit type for debugging
                logger.debug(f"Synthesis response type: {type(response)}, content: {response}")
                if hasattr(response, 'error') and response.error:  # Safely check for error field
                    raise ValueError(response.error)
                final_answer = response.result if hasattr(response, 'result') else str(response)  # Fallback to string
                
                # Build response (unchanged)
                response_parts = ["🤖 **Agent Execution Trace**\n"]
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
            
            # Execute tasks (unchanged)
            if len(tasks) > 1:
                # Concurrent
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
                            # Early return on error (unchanged)
                            response_parts = ["🤖 **Agent Execution Trace**\n"]
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
                # Sequential
                trace = await execute_tool_with_trace(tasks[0]["tool"], tasks[0]["input"], subtask_id)
                execution_traces.append(trace)
                completed_tasks[subtask_id] = trace.output_data
                if trace.status == "error":
                    response_parts = ["🤖 **Agent Execution Trace**\n"]
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
            logger.error(f"Error in agent processing for {subtask_id}: {str(e)}")
            return f"Error in agent processing for {subtask_id}: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")