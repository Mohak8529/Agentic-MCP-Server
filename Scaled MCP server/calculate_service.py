import os
import asyncio
import logging
import math
import grpc
import grpc.aio
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from tools_pb2 import StringRequest, StringResponse
from tools_pb2_grpc import CalculateServiceServicer, add_CalculateServiceServicer_to_server, GroqServiceStub

# Logging (unchanged)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

load_dotenv()

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

GROQ_SERVICE_HOST = "localhost:50051"

class CalculateServicer(CalculateServiceServicer):
    async def Calculate(self, request: StringRequest, context: grpc.aio.ServicerContext):
        max_retries = 3
        current_expression = request.value
        attempted_expressions = set([current_expression])
        
        for attempt in range(max_retries):
            try:
                validated_input = CalculateInput(expression=current_expression)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} validation failed: {str(e)}")
                if attempt < max_retries - 1:
                    refine_prompt = f"""
                    The calculation expression '{current_expression}' failed with error: '{str(e)}'.
                    Previous attempts: {', '.join(attempted_expressions)}.
                    Suggest a corrected mathematical expression using only numbers, operators (+, -, *, /), and valid mathematical functions.
                    Output the refined expression as plain text.
                    """
                    try:
                        async with grpc.aio.insecure_channel(GROQ_SERVICE_HOST) as channel:
                            stub = GroqServiceStub(channel)
                            response = await stub.AskGroq(StringRequest(value=refine_prompt))
                        current_expression = response.result
                        if response.error:
                            raise ValueError(response.error)
                        if current_expression in attempted_expressions:
                            logger.warning(f"Duplicate expression '{current_expression}', skipping")
                            current_expression = request.value
                        else:
                            attempted_expressions.add(current_expression)
                    except Exception as refine_e:
                        logger.error(f"Error refining: {str(refine_e)}")
                        continue
                else:
                    await context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    await context.set_details(str(e))
                    return StringResponse(result="", error=f"Failed after {max_retries} attempts: {str(e)}")
                continue
            
            try:
                safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
                result = eval(validated_input.expression, {"__builtins__": {}}, safe_dict)
                return StringResponse(result=str(result), error="")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} evaluation failed: {str(e)}")
                if attempt < max_retries - 1:
                    refine_prompt = f"""
                    The calculation expression '{current_expression}' failed with error: '{str(e)}'.
                    Previous attempts: {', '.join(attempted_expressions)}.
                    Suggest a corrected mathematical expression using only numbers, operators (+, -, *, /), and valid mathematical functions.
                    Output the refined expression as plain text.
                    """
                    try:
                        async with grpc.aio.insecure_channel(GROQ_SERVICE_HOST) as channel:
                            stub = GroqServiceStub(channel)
                            response = await stub.AskGroq(StringRequest(value=refine_prompt))
                        current_expression = response.result
                        if response.error:
                            raise ValueError(response.error)
                        if current_expression in attempted_expressions:
                            logger.warning(f"Duplicate expression '{current_expression}', skipping")
                            current_expression = request.value
                        else:
                            attempted_expressions.add(current_expression)
                    except Exception as refine_e:
                        logger.error(f"Error refining: {str(refine_e)}")
                        continue
                else:
                    await context.set_code(grpc.StatusCode.INTERNAL)
                    await context.set_details(str(e))
                    return StringResponse(result="", error=f"Failed after {max_retries} attempts: {str(e)}")

async def serve():
    server = grpc.aio.server()
    add_CalculateServiceServicer_to_server(CalculateServicer(), server)
    server.add_insecure_port('[::]:50053')
    logger.info("CalculateService running on port 50053")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    import asyncio
    asyncio.run(serve())