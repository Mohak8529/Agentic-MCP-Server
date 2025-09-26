import os
import asyncio
import logging
import httpx
import re
import grpc
import grpc.aio
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from urllib.parse import quote
from tools_pb2 import StringRequest, StringResponse
from tools_pb2_grpc import WeatherServiceServicer, add_WeatherServiceServicer_to_server, GroqServiceStub

# Logging (unchanged)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

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

GROQ_SERVICE_HOST = "localhost:50051"

class WeatherServicer(WeatherServiceServicer):
    async def GetWeather(self, request: StringRequest, context: grpc.aio.ServicerContext):
        try:
            validated_input = WeatherInput(query=request.value)
        except Exception as e:
            await context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            await context.set_details(str(e))
            return StringResponse(result="", error=str(e))
        
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
                logger.error(f"Attempt {attempt + 1}/{max_retries} geocoding failed: {str(e)}")
                if attempt < max_retries - 1:
                    refine_prompt = f"""
                    The weather query '{current_query}' failed with error: '{str(e)}'.
                    Previous attempts: {', '.join(attempted_queries)}.
                    Suggest a specific location query for the weather API, preferably a city name or city,country.
                    Avoid previously tried queries.
                    Output the refined query as plain text.
                    """
                    try:
                        async with grpc.aio.insecure_channel(GROQ_SERVICE_HOST) as channel:
                            stub = GroqServiceStub(channel)
                            response = await stub.AskGroq(StringRequest(value=refine_prompt))
                        current_query = response.result
                        if response.error:
                            raise ValueError(response.error)
                        if current_query in attempted_queries:
                            logger.warning(f"Duplicate query '{current_query}', skipping")
                            current_query = validated_input.query
                        else:
                            attempted_queries.add(current_query)
                    except Exception as refine_e:
                        logger.error(f"Error refining query: {str(refine_e)}")
                        continue
                else:
                    await context.set_code(grpc.StatusCode.NOT_FOUND)
                    await context.set_details(str(e))
                    return StringResponse(result="", error=f"Failed after {max_retries} attempts: {str(e)}")
                continue
            
            weather_url = "https://api.open-meteo.com/v1/forecast"
            params = {"latitude": latitude, "longitude": longitude, "current": "temperature_2m,weather_code", "timezone": "auto"}
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.get(weather_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                temperature = data["current"]["temperature_2m"]
                weather_code = data["current"]["weather_code"]
                weather_codes = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle", 61: "light rain", 63: "moderate rain", 65: "heavy rain", 71: "light snow", 73: "moderate snow", 75: "heavy snow", 80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers"}
                weather_desc = weather_codes.get(weather_code, "unknown")
                result = f"Current weather in {city}, {country}: {weather_desc}, temperature: {temperature}°C"
                return StringResponse(result=result, error="")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} weather fetch failed: {str(e)}")
                if attempt < max_retries - 1:
                    refine_prompt = f"""
                    The weather query '{current_query}' failed with error: '{str(e)}'.
                    Previous attempts: {', '.join(attempted_queries)}.
                    Suggest a specific location query for the weather API, preferably a city name or city,country.
                    Avoid previously tried queries.
                    Output the refined query as plain text.
                    """
                    try:
                        async with grpc.aio.insecure_channel(GROQ_SERVICE_HOST) as channel:
                            stub = GroqServiceStub(channel)
                            response = await stub.AskGroq(StringRequest(value=refine_prompt))
                        current_query = response.result
                        if response.error:
                            raise ValueError(response.error)
                        if current_query in attempted_queries:
                            logger.warning(f"Duplicate query '{current_query}', skipping")
                            current_query = validated_input.query
                        else:
                            attempted_queries.add(current_query)
                    except Exception as refine_e:
                        logger.error(f"Error refining query: {str(refine_e)}")
                        continue
                else:
                    await context.set_code(grpc.StatusCode.INTERNAL)
                    await context.set_details(str(e))
                    return StringResponse(result="", error=f"Failed after {max_retries} attempts: {str(e)}")

async def serve():
    server = grpc.aio.server()
    add_WeatherServiceServicer_to_server(WeatherServicer(), server)
    server.add_insecure_port('[::]:50052')
    logger.info("WeatherService running on port 50052")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    import asyncio
    asyncio.run(serve())