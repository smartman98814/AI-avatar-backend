"""
Agent Manager for handling 10 OpenAI conversational agents.

This module provides a centralized service for managing OpenAI assistants
in pure conversational mode (no vector store for maximum speed).
"""
import asyncio
from typing import Dict, Optional, List, Any
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from app.config import settings
from app.config.agent_configs import AGENT_CONFIGURATIONS
from app.core import (
    AgentNotFoundError,
    AgentNotInitializedError,
    OpenAIKeyNotConfiguredError,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_THREADPOOL_WORKERS,
    MAX_BUFFER_SIZE,
)
from app.models.agent import AgentConfig, AgentStatus
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AgentManager:
    """
    Manages 10 OpenAI conversational agents (no vector store).
    
    Each agent is an OpenAI assistant in pure conversational mode,
    responding from their system prompts with unique personalities and roles.
    Optimized for maximum speed (2-3s response time).
    """
    
    def __init__(self, max_workers: int = MAX_THREADPOOL_WORKERS):
        """
        Initialize the AgentManager.
        
        Args:
            max_workers: Maximum number of threads for async operations
        """
        self.client: Optional[OpenAI] = None
        self.agents: Dict[int, AgentConfig] = {}
        self.shared_vector_store_id: Optional[str] = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        # Silent initialization for speed
    
    def initialize(self) -> None:
        """
        Initialize the OpenAI client and validate API key.
        
        Raises:
            OpenAIKeyNotConfiguredError: If OpenAI API key is not configured
        """
        if not settings.openai_api_key:
            logger.error("OpenAI API key is not configured")
            raise OpenAIKeyNotConfiguredError()
        
        self.client = OpenAI(api_key=settings.openai_api_key)
        self._initialized = True
    
    
    async def _load_existing_agent(
        self,
        agent_id: int,
        assistant_id: str,
        name: str,
        role: str,
        description: str,
        system_prompt: str,
        model: str
    ) -> AgentConfig:
        """
        Load an existing assistant without vector store (conversational only).
        
        Args:
            agent_id: Unique identifier for the agent (1-10)
            assistant_id: Existing OpenAI assistant ID
            name: Display name of the agent
            role: Role description
            description: Brief description of the agent's purpose
            system_prompt: System prompt for the assistant
            model: OpenAI model to use
            
        Returns:
            AgentConfig object with existing assistant (no vector store)
        """
        self._validate_client_initialized()
        
        loop = asyncio.get_event_loop()
        
        # Retrieve existing assistant
        assistant = await loop.run_in_executor(
            self._executor,
            lambda: self.client.beta.assistants.retrieve(assistant_id)
        )
        
        # Update assistant for conversational mode (no tools, no vector store)
        assistant = await loop.run_in_executor(
            self._executor,
            lambda: self.client.beta.assistants.update(
                assistant_id=assistant_id,
                model=DEFAULT_MODEL,
                temperature=DEFAULT_TEMPERATURE,
                tools=[],  # No tools = faster responses
                tool_resources=None
            )
        )
        
        return AgentConfig(
            id=agent_id,
            name=name,
            role=role,
            description=description,
            system_prompt=system_prompt,
            assistant_id=assistant_id,
            vector_store_id=None,
            model=model
        )
    
    async def create_all_agents(self) -> Dict[int, AgentConfig]:
        """
        Load or create all 10 agents without vector store (conversational mode for speed).
        
        If use_existing_assistants is True and assistant_ids are configured,
        loads existing assistants. Otherwise creates new ones.
        
        Returns:
            Dictionary mapping agent IDs to AgentConfig objects
            
        Raises:
            OpenAIKeyNotConfiguredError: If OpenAI API key is not configured
        """
        if not self._initialized:
            self.initialize()
        
        # No vector store needed - conversational agents only
        self.shared_vector_store_id = None
        
        use_existing = settings.use_existing_assistants
        
        # Load or create all agents
        for agent_id, config in AGENT_CONFIGURATIONS.items():
            try:
                if use_existing and config.get("assistant_id"):
                    # Load existing assistant
                    self.agents[agent_id] = await self._load_existing_agent(
                        agent_id=agent_id,
                        assistant_id=config["assistant_id"],
                        name=config["name"],
                        role=config["role"],
                        description=config["description"],
                        system_prompt=config["system_prompt"],
                        model=config.get("model", DEFAULT_MODEL)
                    )
                else:
                    # Create new assistant
                    self.agents[agent_id] = await self._create_agent(
                        agent_id=agent_id,
                        name=config["name"],
                        role=config["role"],
                        description=config["description"],
                        system_prompt=config["system_prompt"],
                        model=config.get("model", DEFAULT_MODEL)
                    )
            except Exception as e:
                logger.error(f"Error loading/creating agent {agent_id}: {str(e)}", exc_info=True)
                # Create placeholder agent config if loading/creation fails
                self.agents[agent_id] = AgentConfig(
                    id=agent_id,
                    name=config["name"],
                    role=config["role"],
                    description=config["description"],
                    system_prompt=config["system_prompt"],
                    assistant_id=config.get("assistant_id"),
                    vector_store_id=None,
                    model=config.get("model", DEFAULT_MODEL)
                )
        
        loaded = sum(1 for a in self.agents.values() if a.assistant_id)
        logger.info(f"Successfully loaded/created {loaded} agents")
        return self.agents
    
    async def _create_agent(
        self,
        agent_id: int,
        name: str,
        role: str,
        description: str,
        system_prompt: str,
        model: str
    ) -> AgentConfig:
        """
        Create a single conversational agent (no vector store for maximum speed).
        
        Args:
            agent_id: Unique identifier for the agent (1-10)
            name: Display name of the agent
            role: Role enum value
            description: Brief description of the agent's purpose
            system_prompt: System prompt for the assistant (defines personality)
            model: OpenAI model to use
            
        Returns:
            AgentConfig object with assistant (no vector store)
        """
        self._validate_client_initialized()
        
        loop = asyncio.get_event_loop()
        
        # Create assistant for conversational mode (no tools = 3-4s faster!)
        assistant = await loop.run_in_executor(
            self._executor,
            lambda: self.client.beta.assistants.create(
                name=name,
                instructions=system_prompt,
                model=model,
                temperature=DEFAULT_TEMPERATURE,
                tools=[]  # No tools = maximum speed
            )
        )
        
        return AgentConfig(
            id=agent_id,
            name=name,
            role=role,
            description=description,
            system_prompt=system_prompt,
            assistant_id=assistant.id,
            vector_store_id=None,
            model=model
        )
    
    def _validate_client_initialized(self) -> None:
        """Validate OpenAI client is initialized."""
        if not self.client:
            raise OpenAIKeyNotConfiguredError()
    
    def get_agent(self, agent_id: int) -> Optional[AgentConfig]:
        """
        Get agent configuration by ID.
        
        Args:
            agent_id: Agent ID (1-10)
            
        Returns:
            AgentConfig if found, None otherwise
        """
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> Dict[int, AgentConfig]:
        """
        Get all agents.
        
        Returns:
            Dictionary of all agents by ID
        """
        return self.agents
    
    def get_all_agents_status(self) -> List[AgentStatus]:
        """
        Get status of all agents.
        
        Returns:
            List of AgentStatus objects
        """
        return [
            AgentStatus(
                agent_id=agent.id,
                name=agent.name,
                role=agent.role,
                vector_store_id=agent.vector_store_id,
                assistant_id=agent.assistant_id,
                is_ready=bool(agent.assistant_id and agent.vector_store_id)
            )
            for agent in self.agents.values()
        ]
    
    async def chat_with_agent_stream(
        self,
        agent_id: int,
        message: str,
        thread_id: Optional[str] = None,
        buffer_by_sentence: bool = True
    ):
        """
        Chat with an agent using streaming for real-time responses.
        
        Yields response chunks as they're generated for instant feedback.
        Can buffer by sentences/phrases for more readable streaming.
        
        Args:
            agent_id: Agent ID (1-10)
            message: User message to send
            thread_id: Optional thread ID to continue conversation
            buffer_by_sentence: If True, buffers and sends complete sentences/phrases
                              If False, sends individual word chunks (default: True)
            
        Yields:
            JSON chunks containing response text and thread_id
            
        Raises:
            AgentNotFoundError: If agent is not found
            AgentNotInitializedError: If agent is not properly initialized
        """
        agent = self._get_validated_agent(agent_id)
        self._validate_client_initialized()
        
        loop = asyncio.get_event_loop()
        
        # Create or use existing thread
        if not thread_id:
            thread = await loop.run_in_executor(
                self._executor,
                self.client.beta.threads.create
            )
            thread_id = thread.id
        
        # Send thread_id first (using compact keys to reduce payload)
        import json
        yield json.dumps({"t": "s", "tid": thread_id})
        
        # Add message to thread
        await loop.run_in_executor(
            self._executor,
            lambda: self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message
            )
        )
        
        # Create streaming run
        run_stream = await loop.run_in_executor(
            self._executor,
            lambda: self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=agent.assistant_id,
                stream=True
            )
        )
        
        # Buffer for sentence/phrase segmentation
        buffer = ""
        
        # Stream the response chunks
        try:
            for event in run_stream:
                if hasattr(event, 'event') and event.event == 'thread.message.delta':
                    if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                        for content in event.data.delta.content:
                            if hasattr(content, 'text') and hasattr(content.text, 'value'):
                                text_chunk = content.text.value
                                
                                if buffer_by_sentence:
                                    # Add to buffer
                                    buffer += text_chunk
                                    
                                    # Check for sentence/phrase boundaries and yield complete segments
                                    sentences = self._extract_complete_segments(buffer)
                                    for sentence in sentences:
                                        # Compact JSON keys: t=type, d=data, tid=thread_id
                                        yield json.dumps({"t": "c", "d": sentence, "tid": thread_id})
                                    
                                    # Keep incomplete part in buffer
                                    buffer = self._get_remaining_buffer(buffer)
                                    
                                    # Force send if buffer gets too long (ensures fast first response)
                                    if len(buffer) > MAX_BUFFER_SIZE:
                                        yield json.dumps({"t": "c", "d": buffer, "tid": thread_id})
                                        buffer = ""
                                else:
                                    # Send word chunks immediately (compact payload)
                                    yield json.dumps({"t": "c", "d": text_chunk, "tid": thread_id})
                                    
                elif hasattr(event, 'event') and event.event == 'thread.run.completed':
                    # Send any remaining buffer before completion
                    if buffer_by_sentence and buffer.strip():
                        yield json.dumps({"t": "c", "d": buffer, "tid": thread_id})
                    
                    # Signal completion
                    yield json.dumps({"t": "d", "tid": thread_id})
                    break
        except Exception as e:
            yield json.dumps({"t": "e", "e": str(e)})
    
    
    def _get_validated_agent(self, agent_id: int) -> AgentConfig:
        """Get agent and validate it exists and is initialized."""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found")
            raise AgentNotFoundError(agent_id)
        
        if not agent.assistant_id:
            logger.warning(f"Agent {agent_id} is not properly initialized")
            raise AgentNotInitializedError(agent_id)
        
        return agent
    
    def get_shared_vector_store_id(self) -> Optional[str]:
        """Get the shared vector store ID."""
        return self.shared_vector_store_id
    
    def _extract_complete_segments(self, buffer: str) -> List[str]:
        """
        Extract complete sentences/phrases from buffer using optimized approach.
        
        Detects boundaries at: . ! ? , ; : \n
        
        Args:
            buffer: Text buffer to segment
            
        Returns:
            List of complete segments ready to yield
        """
        segments = []
        last_idx = 0
        
        # Fast scan for boundaries (avoid regex overhead)
        for i, char in enumerate(buffer):
            if char in '.!?,;:\n':
                # Check if followed by space or end (to avoid splitting decimals like 3.14)
                if i + 1 < len(buffer) and buffer[i + 1] in ' \n\t':
                    segments.append(buffer[last_idx:i + 2])  # Include boundary + space
                    last_idx = i + 2
                elif i + 1 >= len(buffer):
                    segments.append(buffer[last_idx:i + 1])
                    last_idx = i + 1
        
        return segments
    
    def _get_remaining_buffer(self, buffer: str) -> str:
        """
        Get the remaining incomplete text from buffer (optimized).
        
        Args:
            buffer: Current buffer
            
        Returns:
            Remaining incomplete text that hasn't been yielded
        """
        # Fast reverse scan for last boundary
        for i in range(len(buffer) - 1, -1, -1):
            if buffer[i] in '.!?,;:\n':
                if i + 1 < len(buffer) and buffer[i + 1] in ' \n\t':
                    return buffer[i + 2:]
                elif i + 1 >= len(buffer):
                    return ""
        
        return buffer
    
    

# Global agent manager instance
agent_manager = AgentManager()
