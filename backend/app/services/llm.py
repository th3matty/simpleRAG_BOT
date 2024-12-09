from anthropic import Anthropic
from typing import List, Dict, Any
from ..exceptions import LLMError
from ..config import settings, logger
from .tools import TOOLS, ToolExecutor


class LLMService:
    _instance = None

    def __new__(cls, api_key: str):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            try:
                logger.info("Initializing Anthropic client")
                cls._instance.client = Anthropic(api_key=api_key)
                cls._instance.tools = TOOLS
                logger.info(f"Initialized tools: {[tool['name'] for tool in TOOLS]}")
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
            logger.warning(
                f"Context length ({len(combined_context)}) exceeds maximum ({settings.max_context_length}). Truncating..."
            )
            return combined_context[: settings.max_context_length]
        return combined_context

    def process_query(
        self, query: str, tool_executor: ToolExecutor, model: str = None
    ) -> Dict[str, Any]:
        """
        Process a user query, potentially using tools to find information.

        Args:
            query: User query string
            tool_executor: ToolExecutor instance for handling tool calls
            model: Optional model override

        Returns:
            Dictionary containing response text and metadata
        """
        try:
            # Use configured model if none provided
            model = model or settings.model_name

            # Initial system prompt for tool usage
            system_prompt = """You are a helpful assistant that can use tools to find information and perform calculations.

Instructions for tool usage:
1. When a user asks a question, decide if you need to use any tools
2. For information queries, use the search_documents tool
3. For mathematical calculations, use the calculator tool
4. After getting tool results, provide a focused answer
5. If you don't need any tools, respond directly

Remember:
- Only use tools when necessary
- Keep tool inputs focused and relevant
- Use tool results to provide accurate, specific answers"""

            # First interaction - decide if tools are needed
            messages = [{"role": "user", "content": query}]

            logger.debug(f"Sending initial request to Claude with model: {model}")
            response = self.client.messages.create(
                model=model,
                system=system_prompt,
                messages=messages,
                tools=self.tools,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

            logger.debug(f"Claude response type: {response.content[0].type}")

            # Check if tool use is requested
            for content_block in response.content:
                if content_block.type == "tool_use":
                    # Execute the requested tool
                    tool_name = content_block.name
                    tool_input = content_block.input
                    tool_use_id = content_block.id

                    logger.info(
                        f"Tool use requested: {tool_name} with input: {tool_input}"
                    )

                    # Execute the tool and get result
                    tool_result = tool_executor.execute_tool(tool_name, tool_input)

                    # Generate final response with tool result
                    final_response = self.generate_response(
                        query=query,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_result=tool_result,
                        tool_use_id=tool_use_id,
                        model=model,
                    )

                    return {
                        "text": final_response["text"],
                        "model": final_response["model"],
                        "finish_reason": final_response["finish_reason"],
                        "usage": final_response["usage"],
                        "tool_used": tool_name,
                        "tool_input": tool_input,
                        "tool_result": tool_result,
                    }

                elif content_block.type == "text":
                    if content_block.text.startswith("<thinking>"):
                        logger.debug(f"Thinking block: {content_block.text}")
                        continue  # Skip thinking blocks
                    # Direct text response
                    logger.info("Direct text response without tool use")
                    return {
                        "text": content_block.text,
                        "model": model,
                        "finish_reason": response.stop_reason,
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        },
                        "tool_used": None,
                    }

            raise LLMError("No valid response content found")

        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            raise LLMError(f"Failed to process query: {str(e)}")

    def generate_response(
        self,
        query: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: str,
        tool_use_id: str,
        model: str = None,
    ) -> Dict[str, Any]:
        """
        Generate a final response using tool results.

        Args:
            query: Original user query string
            tool_name: Name of the tool that was used
            tool_input: Input that was provided to the tool
            tool_result: Result returned by the tool
            tool_use_id: The ID of the tool use request
            model: Optional model override

        Returns:
            Dictionary containing response text and metadata

        Raises:
            LLMError: If response generation fails
        """
        try:
            # Use configured model if none provided
            model = model or settings.model_name

            # System prompt for final response
            system_prompt = """You are a helpful assistant that provides answers based on tool results.

Instructions:
1. Use the tool result to answer the original question
2. Explain the result in a clear and natural way
3. If the tool result indicates an error, explain what went wrong
4. Keep your response focused and concise

Remember: Stay focused on answering the specific question using the tool results."""

            # Format the messages including the tool result
            messages = [
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tool_name,
                            "input": tool_input,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_result,
                        }
                    ],
                },
            ]

            logger.debug(f"Sending final request to Claude with model: {model}")
            logger.debug(f"Tool result: {tool_result}")

            response = self.client.messages.create(
                model=model,
                system=system_prompt,
                messages=messages,
                tools=self.tools,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

            # Extract response text and metadata
            result = {
                "text": response.content[0].text,
                "model": model,
                "finish_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

            logger.info(
                f"Generated final response with {result['usage']['output_tokens']} tokens"
            )
            logger.debug(
                f"Final response: {result['text'][:100]}..."
            )  # Log first 100 chars
            return result

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")
