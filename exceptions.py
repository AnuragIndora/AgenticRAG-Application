class AgenticRAGError(Exception):
    """Base class for application-specific exceptions."""


class UnsupportedFileFormatError(AgenticRAGError):
    """Raised when ingestion receives an unsupported file type."""


class MilvusConnectionError(AgenticRAGError):
    """Raised when Milvus connectivity fails."""


class EmptyRetrievalError(AgenticRAGError):
    """Raised when retrieval does not return any usable context."""


class OllamaTimeoutError(AgenticRAGError):
    """Raised when Ollama call times out."""
