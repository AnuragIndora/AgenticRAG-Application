from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import shutil
import logging
import pandas as pd

from real_time_agent import RealTimeAgentAssistant

app = FastAPI(title="Agentic RAG System API")
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = Path("uploaded_docs")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# The assistant starts a background thread to watch UPLOAD_FOLDER for new files
assistant = RealTimeAgentAssistant(watch_folder=str(UPLOAD_FOLDER))


class QueryRequest(BaseModel):
    query: str
    task_type: str | None = None


@app.post("/query")
def query_agent(request: QueryRequest):
    try:
        result = assistant.query(request.query, task_type=request.task_type)
        return {
            "answer": result.answer,
            "intent": result.intent,
            "confidence_score": result.confidence_score,
            "sources": [s.model_dump() for s in result.sources],
            "reasoning_steps": result.reasoning_steps,
            "sql_query": result.sql_query,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    """Return the list of files that have already been ingested."""
    return {"ingested_files": list(assistant.ingested_files)}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".docx", ".pptx", ".txt", ".md", ".xlsx", ".csv"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    dest = UPLOAD_FOLDER / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Structured files (CSV / Excel) go straight into Postgres, not Milvus
    if suffix in {".xlsx", ".csv"}:
        from postgres_client import PostgresClient

        db = PostgresClient()
        table_name = file.filename.split(".")[0].lower()

        if suffix == ".csv":
            df = pd.read_csv(dest)
        else:
            df = pd.read_excel(dest)

        db.create_table_from_dataframe(table_name, df)
        dest.unlink()  # remove the temp file — the data now lives in Postgres

        return {"message": f"Structured data loaded into table '{table_name}'"}

    # Unstructured files are picked up automatically by the folder watcher
    return {"message": "File uploaded for vector ingestion."}


@app.delete("/reset")
def reset_database(delete_sql: bool = False):
    """
    Wipe all ingested data from Milvus.
    Pass `?delete_sql=true` to also drop all Postgres tables.
    """
    try:
        assistant.orchestrator.ingestion_pipeline.milvus.delete_collection()
        assistant.ingested_files.clear()

        if delete_sql:
            assistant.orchestrator.structured_agent.sql_agent.delete_all_tables()

        return {"message": "Database and ingested files cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)