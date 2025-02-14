from ..core.exceptions import LLMError
from ..core.config import settings
import logging
from langdetect import detect  # Add langdetect import

logger = logging.getLogger(__name__)

from anthropic import Anthropic
from typing import List, Dict, Any
from ..core.tools import TOOLS, ToolExecutor


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
            # Detect query language
            try:
                detected_lang = detect(query)
                logger.info(f"Detected query language: {detected_lang}")
            except:
                detected_lang = "en"
                logger.warning("Failed to detect language, defaulting to English")

            # Use configured model if none provided
            model = model or settings.model_name

            # Enhanced system prompt for better RAG responses
            system_prompt = f"""You are a helpful assistant with access to a knowledge base of documents. Your primary role is to provide accurate, evidence-based answers by ALWAYS searching through these documents first.

CRITICAL INSTRUCTIONS:
1. ALWAYS use search_documents tool FIRST before any response
2. After finding information, you MUST respond in {detected_lang} language
3. If query is in German, respond in German
4. If query is in English, respond in English
5. Match the language of the user's query exactly

Core Principles:
1. ASSUME ALL USER QUERIES MIGHT HAVE RELEVANT INFORMATION IN THE DOCUMENTS
2. ALWAYS search documents BEFORE attempting to answer any factual question
3. Only provide information that can be supported by the documents
4. Format response in the user's language ({detected_lang})

Instructions for Every Query:

1. Initial Document Search (MANDATORY):
   - For EVERY query, you MUST FIRST use the search_documents tool
   - Break complex queries into multiple focused searches
   - Use variations of key terms to ensure comprehensive coverage
   - For general queries, use broader search terms to discover available information

2. Search Strategy:
   - Start with specific, focused search terms
   - If initial search yields low relevance, try alternative phrasings
   - For multi-part questions, conduct separate searches for each part
   - Consider synonyms and related terms to broaden search coverage

3. Analyzing Search Results:
   - Evaluate relevance scores (High, Moderate, Low) critically
   - Only use information from documents, not general knowledge
   - Cross-reference information across multiple documents
   - If relevance scores are low, acknowledge limited document coverage

4. Response Formulation:
   - Begin with "Basierend auf den verfügbaren Dokumenten..." for German queries
   - Begin with "Based on the available documents..." for English queries
   - Only make statements that are directly supported by documents
   - Always cite specific document references with relevance scores
   - Clearly indicate when information is not found in documents
   - Synthesize information from multiple documents when relevant
   - ENSURE response is in {detected_lang} language

5. Tool Usage:
   - Use search_documents tool as your primary information source
   - Combine multiple searches for comprehensive coverage

Response Structure:
1. Document-based answer (in {detected_lang})
2. Specific document citations with relevance scores
3. Calculations (if needed) using document data
4. Clear statement about information coverage or gaps

Remember:
- ALWAYS use search_documents tool before responding
- Never assume you know the answer without searching
- Always cite your document sources
- Be explicit about information not found in documents
- Maintain transparency about search comprehensiveness
- YOU MUST RESPOND IN {detected_lang.upper()} LANGUAGE"""

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
                        detected_lang=detected_lang,  # Pass the detected language
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
        detected_lang: str = "en",  # Add detected_lang parameter with default to English
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
            detected_lang: The detected language of the query (defaults to English)

        Returns:
            Dictionary containing response text and metadata

        Raises:
            LLMError: If response generation fails
        """
        try:
            # Use configured model if none provided
            model = model or settings.model_name

            # Enhanced system prompt for final response generation
            system_prompt = f"""You are a helpful assistant that generates responses based STRICTLY on document search results and tool outputs.

CRITICAL INSTRUCTIONS:
1. YOU MUST RESPOND IN {detected_lang.upper()} LANGUAGE
2. If query is in German, respond in German
3. If query is in English, respond in English
4. Match the language of the user's query exactly

Core Principles:
1. Use ONLY information found in the searched documents
2. Begin with "Basierend auf den verfügbaren Dokumenten..." for German queries
3. Begin with "Based on the available documents..." for English queries
4. Every claim must have a document reference
5. Format response in the user's language ({detected_lang})

Instructions for Response Generation:

1. Document Analysis:
   - Analyze search results with their relevance scores critically
   - Focus on highly relevant documents first
   - Cross-reference information across documents
   - Identify and acknowledge information gaps
   - Consider completeness of document coverage

2. Response Structure:
   - Begin with appropriate language prefix based on {detected_lang}
   - Include specific document references for EVERY claim
   - Always cite relevance scores with document references
   - Clearly state when information is not found
   - Maintain logical organization of information
   - ENSURE response is in {detected_lang} language

3. Quality Control:
   - Verify each statement has document support
   - Cross-validate information across documents
   - Highlight any inconsistencies found
   - Be explicit about confidence levels
   - Never include general knowledge without document support

4. Tool Result Integration:
   - Incorporate search results comprehensively
   - Show calculation steps with document-based data
   - Explain any search limitations or gaps
   - Suggest additional searches if needed
   - Document all data sources used

Remember:
- Never assume information exists without finding it
- Always provide document references
- Be transparent about search limitations
- Maintain strict evidence-based responses
- YOU MUST RESPOND IN {detected_lang.upper()} LANGUAGE"""

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
