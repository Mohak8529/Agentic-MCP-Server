import asyncio
import sys
from fastmcp import Client

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py 'your question here'")
        return

    user_question = sys.argv[1]

    # Initialize the client with the path to your server script
    async with Client(transport="server.py") as client:
        # Call the 'agent' tool with the user's question
        result = await client.call_tool("agent", {"task": user_question})
        
        # Access the result content properly
        print("\nAnswer:", result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())