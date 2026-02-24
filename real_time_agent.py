import time
from pathlib import Path
import threading
import logging

from agents_pipeline import AgentsOrchestrator

logger = logging.getLogger("RealTimeAgent")
logging.basicConfig(level=logging.INFO)


class RealTimeAgentAssistant:
    """
    Watches a folder for new files and ingests them automatically in the background.
    Also provides a `query()` interface to the agent orchestrator.
    """

    def __init__(self, watch_folder: str, poll_interval: int = 5):
        self.watch_folder = Path(watch_folder)
        if not self.watch_folder.is_dir():
            raise ValueError(f"Folder does not exist: {watch_folder}")

        self.poll_interval = poll_interval  # seconds between directory scans
        self.orchestrator = AgentsOrchestrator()

        # Snapshot existing files so we don't re-ingest them on startup
        self.ingested_files = set(f.name for f in self.watch_folder.iterdir() if f.is_file())

        # Run the watcher as a daemon so it dies when the main process exits
        self._watcher_thread = threading.Thread(target=self._watch_folder, daemon=True)
        self._watcher_thread.start()

    def _watch_folder(self):
        """Background loop: detect new files and trigger ingestion."""
        logger.info(f"Started watching folder: {self.watch_folder}")
        while True:
            try:
                current_files = set(f.name for f in self.watch_folder.iterdir() if f.is_file())
                new_files = current_files - self.ingested_files

                for file_name in new_files:
                    file_path = self.watch_folder / file_name
                    logger.info(f"New file detected: {file_name} — ingesting...")
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

    def query(self, text: str, task_type=None):
        """Forward a query to the orchestrator and return the result."""
        return self.orchestrator.handle_query(text, task_type)