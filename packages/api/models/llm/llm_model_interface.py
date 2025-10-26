from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable

from ...domain.memory_context import MemoryContext
from ...domain.memory_request import MemoryRequest


class MemoryResponse:
    """
    Represents a response from an LLM model.

    Encapsulates the generated text along with metadata about the generation
    process, including token usage and model information.
    """

    def __init__(
        self,
        response: MemoryRequest,
        model_name: str,
        tokens_used: int,
        metadata: dict = None,
    ):
        """
        Initialize a memory response.

        Args:
            response: The generated response as a MemoryRequest object
            model_name: Name of the model that generated the response
            tokens_used: Number of tokens consumed during generation
            metadata: Additional metadata about the generation (optional)
        """
        self.model_name = model_name
        self.tokens_used = tokens_used
        self.metadata = metadata or {}
        self.response = response


class LLMModel(ABC):
    """
    Abstract base class for large language models that generate text responses.
    """

    @abstractmethod
    async def generate_with_memory(
        self,
        prompt: str,
        memory_context: MemoryContext,
        chunk_size_tokens: int = 1,
    ) -> AsyncIterator[MemoryResponse]:
        """
        Generate a text response based on the prompt and conversation history.
        Returns an async iterator that yields chunks of the response.

        Args:
            prompt: The user prompt to generate a response for
            memory_context: Memory context object containing relevant memories
                (from vector search)
            chunk_size_tokens: Number of tokens to coalesce per streaming chunk.
                Higher values reduce object churn but increase latency between
                visible updates.
                Setting a very large value effectively creates non-streaming behavior.
        Returns:
            AsyncIterator[MemoryResponse]: Stream of response chunks
        """
        pass


class LLMModelBase(LLMModel, ABC):
    """
    Base class for LLM models with common prompt formatting and memory handling.
    Provides default implementations for memory formatting and message building.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access 
to the user's memories. When answering questions, ONLY use information 
provided in the context. If the answer cannot be found in the context, 
say "I don't have any memories about that." Be concise and focus only on 
information found in the relevant memories. Always cite the source of the 
information using the format <source>memory.id</source> at the end of a sentence.
Make sure to always answer in second person. NEVER say: "I had ..." in first 
person. Always answer in second person!

<examples>
These are only illustrations. Do NOT treat them as actual memories. 
They are here to demonstrate the answering format.

    <example>
        <system_prompt>
        Pretend this is what a memory input looks like in 
        JSON format: {'id': '17424128-3e76-4aa3-8230-aeaae77385e0', 
        'score': 0.83, 'content': 'In the year 1723 on first of december i
          had a toast for breakfast'}
          </system_prompt>
        <users_prompt>
        What did i eat in the morning 
        of the 1st of december 1723</users_prompt>
        <your_answer>
        You had a toast for breakfast on the 
        1st of december 1723 <source>17424128-3e76-4aa3-8230-aeaae77385e0</source>
        </your_answer>
    </example>
    <example>
        <system_prompt>
        Pretend this is what a memory input looks like in JSON format: 
        {'id': '17424128-3e76-4aa3-8230-aeaae77385e0', 'score': 0.83, 
        'content': 'In the year 1723 on first of december 
        i had a toast for breakfast'}
        </system_prompt>
        <users_prompt>What is 1 + 1</users_prompt>
        <your_answer>I don't have any memories about that.</your_answer>
    </example>

Ignore all example blocks when generating an answer. 
Only use memory JSONs that appear after this point and outside of <examples>
</examples>

/nothink
"""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        top_k_memories: int = 5,
    ):
        """
        Initialize the base LLM model.

        Args:
            system_prompt: Custom system prompt. If None, uses DEFAULT_SYSTEM_PROMPT.
            top_k_memories: Number of top memories to include in context.
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.top_k_memories = top_k_memories

    def _format_memories(self, top: Iterable[tuple], max_chars: int = 1200) -> str:
        """
        Format memories as a JSON-like structure for better LLM comprehension.

        Args:
            top: Iterable of (memory, matched_text, score) tuples
            max_chars: Maximum characters per memory content

        Returns:
            Formatted string of memories in JSON-like format
        """
        memories = []
        for mem, matched_text, score in top:
            snippet = (matched_text or "").strip()
            if max_chars and len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3] + "..."

            memories.append(
                f'{{ "id": "{mem.id}", "score": {score:.4f}, "content": "{snippet}" }}'
            )

        if not memories:
            return "No relevant memories found."

        return "\n".join(memories)

    def build_messages(
        self, prompt: str, memory_context: MemoryContext | None
    ) -> list[dict[str, str]]:
        """
        Build messages for the chat completion API using the provided prompt
        and memory context. Format the context in a way that's easier for
        small models to understand.

        Args:
            prompt: User's question or prompt
            memory_context: Optional memory context containing relevant memories

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        if memory_context:
            top = memory_context.get_top_memories(limit=self.top_k_memories)

            # Format context with JSON-like memory structures
            context_parts = [
                "Here are relevant memories in JSON format:<memories>",
                self._format_memories(top, max_chars=1200),
                "</memories>\nAnswer the question using ONLY the <memories> content!",
            ]

            messages.append({
                "role": "system",
                "content": "\n\n".join(context_parts),
            })

        messages.append({"role": "user", "content": prompt})
        return messages
