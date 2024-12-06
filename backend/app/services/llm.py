from anthropic import Anthropic
from typing import List, Dict, Any
from ..exceptions import LLMError
from ..config import settings, logger

class LLMService:
    _instance = None

    def __new__(cls, api_key: str):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            try:
                logger.info("Initializing Anthropic client")
                cls._instance.client = Anthropic(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")
                raise LLMError(f"Failed to initialize Anthropic client: {str(e)}")
        return cls._instance

    def _prepare_context(self, context: List[str]) -> str:
        """
        Prepare context for the LLM by joining and truncating if necessary.
        
        Args:
            context: List of context strings
            
        Returns:
            Formatted context string
        """
        combined_context = " ".join(context)
        if len(combined_context) > settings.max_context_length:
            logger.warning(f"Context length ({len(combined_context)}) exceeds maximum ({settings.max_context_length}). Truncating...")
            return combined_context[:settings.max_context_length]
        return combined_context

    def generate_response(self, 
                         query: str, 
                         context: List[str],
                         model: str = None) -> Dict[str, Any]:
        """
        Generate a response using Claude.
        
        Args:
            query: User query string
            context: List of context strings
            model: Optional model override
            
        Returns:
            Dictionary containing response text and metadata
            
        Raises:
            LLMError: If response generation fails
        """
        try:
            # Use configured model if none provided
            model = model or settings.model_name
            
            # Prepare system prompt and context
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
1. Answer ONLY what was specifically asked in the question
2. Use ONLY the information from the context that is directly relevant to the question
3. If you find information in the context that isn't relevant to the question, ignore it completely
4. Keep your response focused and concise
5. If you can't find relevant information to answer the question, simply state that you don't have enough information

Remember: Stay strictly focused on answering the specific question asked using only relevant information."""
            
            formatted_context = self._prepare_context(context)
            
            # Format the message for Claude
            messages = [
                {
                    "role": "user",
                    "content": f"Context: {formatted_context}\n\nQuestion: {query}"
                }
            ]
            
            logger.debug(f"Sending request to Claude with model: {model}")
            response = self.client.messages.create(
                model=model,
                system=system_prompt,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
            
            # Extract response text and metadata
            result = {
                "text": response.content[0].text,
                "model": model,
                "finish_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
            logger.info(f"Generated response with {result['usage']['output_tokens']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")
