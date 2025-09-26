import os
import logging
import httpx
import grpc
import grpc.aio
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from tools_pb2 import StringRequest, StringResponse
from tools_pb2_grpc import GroqServiceServicer, add_GroqServiceServicer_to_server

# Logging setup (unchanged)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY environment variable")

class GroqInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="Prompt for Groq API")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or just whitespace")
        return v.strip()

class GroqServicer(GroqServiceServicer):
    async def AskGroq(self, request: StringRequest, context: grpc.aio.ServicerContext):
        try:
            validated_input = GroqInput(prompt=request.value)
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "meta-llama/llama-4-scout-17b-16e-instruct", "messages": [{"role": "user", "content": validated_input.prompt}]}
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            content = data["choices"][0]["message"]["content"]
            return StringResponse(result=content, error="")
        except Exception as e:
            logger.error(f"Error in AskGroq: {str(e)}")
            await context.set_code(grpc.StatusCode.INTERNAL)
            await context.set_details(str(e))
            return StringResponse(result="", error=str(e))

async def serve():
    server = grpc.aio.server()
    add_GroqServiceServicer_to_server(GroqServicer(), server)
    server.add_insecure_port('[::]:50051')
    logger.info("GroqService running on port 50051")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    import asyncio
    asyncio.run(serve())