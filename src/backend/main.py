import os
from pathlib import Path
from typing import Any, Dict, List
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
#from langchain_core.retrievers import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from load_data import load_all_from_folder
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Simple React Backend")




@app.post("/api/research")
async def do_rag(data: Dict[str, str]) -> Dict[str, str]:
    # Путь к сохранённому индексу
    faiss_path = "backend/faiss_index"

    # ВАЖНО: тот же тип эмбеддингов, что и при создании индекса
    embeddings = OpenAIEmbeddings()

    # Загрузка FAISS из локальной папки
    db = FAISS.load_local(folder_path=faiss_path, embeddings=embeddings,
                          allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 3})  # k = сколько результатов вернуть

    template = """You are the best documentation reader and explanator.
    Use the context to answer the research query from the user. 
    If you do not have enough information, answer that you need more documentation. 
    
    Context:
    {context}
    
    Research query: {input}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    # llm = init_chat_model("gpt-3.5-turbo-0125")
    #
    # question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # print('data ', data)
    # qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    qa_chain = create_retrieval_chain(retriever, prompt | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0))


    async def generate(query):
        async for chunk in qa_chain.astream({"input": query['request']}):
            print('chunk!', chunk)
            text = chunk.get("answer", "")
            yield f"data: {text}\n\n"


    #return {"message": "Everything is fine!"}
    to_return = StreamingResponse(
        content=generate(data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )
    return to_return


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



    return {"uploaded": to_return}