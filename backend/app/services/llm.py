from anthropic import Anthropic
from typing import List

class LLMService:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        
    def generate_response(self, 
                         query: str, 
                         context: List[str],
                         model: str = "claude-3-sonnet-20240229") -> str:
        """Generate a response using Claude."""
        system_prompt = "You are a helpful assistant. Use the provided context to answer questions."
        
        # Format the message for Claude
        messages = [
            {
                "role": "user",
                "content": f"Context: {' '.join(context)}\n\nQuestion: {query}"
            }
        ]
        
        response = self.client.messages.create(
            model=model,
            system=system_prompt,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.content[0].text