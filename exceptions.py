class AgenticRAGError(Exception):
    """Base class for all application-specific exceptions."""


class UnsupportedFileFormatError(AgenticRAGError):
    """Raised when an unsupported file type is passed to the ingestion pipeline."""


class MilvusConnectionError(AgenticRAGError):
    """Raised when the app can't connect to or communicate with Milvus."""


class EmptyRetrievalError(AgenticRAGError):
    """Raised when a vector search returns no usable results."""


class OllamaTimeoutError(AgenticRAGError):
    """Raised when a request to the Ollama API times out or returns an empty response."""
