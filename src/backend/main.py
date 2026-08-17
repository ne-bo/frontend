import os
from pathlib import Path
from typing import Any, Dict, List
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from load_data import load_all_from_folder
from langchain_community.vectorstores import FAISS

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
    docs = []
    for f in files:
        safe_name = f.filename.replace("\\", "/").split("/")[-1]
        if not safe_name:
            raise HTTPException(status_code=400, detail="Empty filename")

        file_path = upload_dir / safe_name
        with open(file_path, "wb") as out_file:
            out_file.write(await f.read())

        saved_paths.append({"filename": safe_name, "path": str(file_path.absolute())})
        to_return.append({
            "name": safe_name,
            "size": f.size,
            "type": f.content_type
        })

    docs = load_all_from_folder("uploads")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    print('docs ', docs)
    splitted_docs = text_splitter.split_documents(docs)
    print('splitted docs ', splitted_docs)

    embeddings = OpenAIEmbeddings()
    print('embeddings ', embeddings)

    faiss_index_path = "backend/faiss_index"
    os.makedirs(faiss_index_path, exist_ok=True)


    db = FAISS.from_documents(
        documents=splitted_docs,
        embedding=embeddings
    )

    db.save_local(faiss_index_path)

    # model = init_chat_model("gpt-3.5-turbo-0125")
    # result = model.invoke("Hello, world!")
    # print('result ', result)

    return {"uploaded": to_return}