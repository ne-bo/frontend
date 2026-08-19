import os
import io
from pathlib import Path
from typing import (Any,
                    Dict,
                    List)
from dotenv import load_dotenv

from langchain_openai import (OpenAIEmbeddings,
                              ChatOpenAI)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from fastapi import (FastAPI,
                     HTTPException,
                     UploadFile,
                     File)
from fastapi.responses import StreamingResponse

from constants import (FAISS_PATH,
                       PROMPT_TEMPLATE,
                       OPENAI_MODEL_NAME,
                       TEMPERATURE,
                       CHUNKS_TO_RETRIVE)
from utils import (compute_content_hash,
                   load_processed_hashes,
                   save_processed_hashes,
                   add_documents_to_index,
                   add_file_to_the_documentation)

load_dotenv()

app = FastAPI(title="RAG Backend")


def initialize_the_index():
    global embeddings, db, retriever, prompt, llm, combine_docs_chain, qa_chain

    embeddings = OpenAIEmbeddings()

    if os.path.exists(FAISS_PATH):
        db = FAISS.load_local(
            folder_path=FAISS_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": CHUNKS_TO_RETRIVE})
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    else:
        db = None
        retriever = None
        combine_docs_chain = None
        qa_chain = None


    prompt = ChatPromptTemplate.from_template(template=PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=TEMPERATURE)




async def update_the_index():
    global db, retriever, combine_docs_chain, qa_chain

    db = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": CHUNKS_TO_RETRIVE})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)


initialize_the_index()


@app.post("/api/research")
async def do_rag(data: dict):

    if qa_chain is None:
        raise HTTPException(
        status_code=503,
        detail="Upload the documents first"
        )


    async def generate(query):
        async for chunk in qa_chain.astream({"input": query['request']}):
            text = chunk.get("answer", "")
            yield text

    to_return = StreamingResponse(
        content=generate(data),
        media_type="text/markdown",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )
    return to_return


@app.post("/api/sources")
async def upload_files(files: List[UploadFile] = File(...)):
    processed_hashes = load_processed_hashes()
    to_return = []
    docs = []
    for f in files:
        safe_name = f.filename.replace("\\", "/").split("/")[-1]
        if not safe_name:
            raise HTTPException(status_code=400, detail="Empty filename")
        extension = f.filename.lower().split('.')[-1]
        content_bytes = await f.read()
        file_hash = compute_content_hash(content_bytes)

        if file_hash not in processed_hashes:
            await add_file_to_the_documentation(content_bytes,
                                                docs,
                                                extension,
                                                file_hash,
                                                processed_hashes,
                                                safe_name)

        to_return.append({
            "name": safe_name,
            "size": f.size,
            "type": f.content_type
        })
    save_processed_hashes(processed_hashes)

    if len(docs) > 0:
        await add_documents_to_index(docs)

        await update_the_index()

    return {"uploaded": to_return}
