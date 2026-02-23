# real_time_agent.py
import time
from pathlib import Path
import threading
import logging

from agents_pipeline import AgentsOrchestrator

logger = logging.getLogger("RealTimeAgent")
logging.basicConfig(level=logging.INFO)


class RealTimeAgentAssistant:
    """
    Watches a folder for new files, automatically ingests them,
    and provides an interface to query agents.
    """
    def __init__(self, watch_folder: str, poll_interval: int = 5):
        self.watch_folder = Path(watch_folder)
        if not self.watch_folder.is_dir():
            raise ValueError(f"Folder does not exist: {watch_folder}")
        self.poll_interval = poll_interval
        self.orchestrator = AgentsOrchestrator()
        self.ingested_files = set(f.name for f in self.watch_folder.iterdir() if f.is_file())

        # Start watcher in a separate thread
        self._watcher_thread = threading.Thread(target=self._watch_folder, daemon=True)
        self._watcher_thread.start()

    def _watch_folder(self):
        """Internal loop to watch folder and ingest new files."""
        logger.info(f"Started watching folder: {self.watch_folder}")
        while True:
            try:
                current_files = set(f.name for f in self.watch_folder.iterdir() if f.is_file())
                new_files = current_files - self.ingested_files
                for file_name in new_files:
                    file_path = self.watch_folder / file_name
                    logger.info(f"Detected new file: {file_name}, ingesting...")
                    try:
                        self.orchestrator.ingest_file(str(file_path))
                        self.ingested_files.add(file_name)
                        logger.info(f"Successfully ingested: {file_name}")
                    except Exception as e:
                        logger.error(f"Failed to ingest {file_name}: {e}")
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in folder watcher: {e}")
                time.sleep(self.poll_interval)

    def query(self, text: str, task_type=None) -> str:
        """Query the agent via orchestrator."""
        return self.orchestrator.handle_query(text, task_type)


# Example usage:
# assistant = RealTimeAgentAssistant(watch_folder="example_docs")
# answer = assistant.query("Explain the ingestion pipeline.")
# print(answer)