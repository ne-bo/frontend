from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

def load_all_from_folder(folder_path: str):
    folder = Path(folder_path)
    documents = []

    # .txt
    for f in folder.glob("*"):
        if f.is_file() and f.suffix.lower() not in [".pdf", ".doc", ".docx"]:
            loader = TextLoader(str(f), encoding="utf-8")
            for d in loader.load():
                d.metadata["source"] = str(f)
                documents.append(d)


    # .pdf
    for f in folder.glob("*.pdf"):
        loader = PyPDFLoader(str(f))
        for d in loader.load():
            d.metadata["source"] = str(f)
            documents.append(d)

    # .docx
    for f in folder.glob("*.doc*"):
        loader = Docx2txtLoader(str(f))
        for d in loader.load():
            d.metadata["source"] = str(f)
            documents.append(d)

    return documents


