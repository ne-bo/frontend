import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Simple React Backend")


@app.post("/api/research")
def do_rag(data: Dict[str, str]) -> Dict[str, str]:
    return {"message": "Everything is fine!"}


@app.post("/api/sources")
async def upload_files(files: List[UploadFile] = File(...)) -> Dict[str, List[Dict[str, Any]]]:
    upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    to_return = []
    for f in files:
        safe_name = f.filename.replace("\\", "/").split("/")[-1]
        if not safe_name:
            raise HTTPException(status_code=400, detail="Empty filename")

        file_path = upload_dir / safe_name
        with open(file_path, "wb") as buffer:
            buffer.write(await f.read())

        saved_paths.append({"filename": safe_name, "path": str(file_path.absolute())})
        to_return.append({
            "name": safe_name,
            "size": f.size,
            "type": f.content_type
        })

    return {"uploaded": to_return}