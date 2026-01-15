from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
from typing import List, Tuple, Union
import asyncio
from collections import deque
from functools import lru_cache
import pickle
import os

from flashrag.config import Config
from flashrag.utils import get_retriever

app = FastAPI()

retriever_list = []
available_retrievers = deque()
retriever_semaphore = None

CACHE_FILE = "retriever_cache.pkl"
CACHE_SAVE_INTERVAL = 100
request_count = 0

# Load cache if exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        retriever_cache = pickle.load(f)
else:
    retriever_cache = {}

def init_retriever(args):
    global retriever_semaphore
    config = Config(args.config)
    for i in range(args.num_retriever):
        print(f"Initializing retriever {i+1}/{args.num_retriever}")
        retriever = get_retriever(config)
        retriever_list.append(retriever)
        available_retrievers.append(i)
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        }
    }

class QueryRequest(BaseModel):
    query: str
    top_n: int = 10
    return_score: bool = False

class BatchQueryRequest(BaseModel):
    query: List[str]
    top_n: int = 10
    return_score: bool = False

class Document(BaseModel):
    id: str
    contents: str

@lru_cache(maxsize=1000)
def cached_search(retriever_idx, query, top_n, return_score):
    return retriever_list[retriever_idx].search(query, top_n, return_score)

@app.post("/search", response_model=Union[Tuple[List[Document], List[float]], List[Document]])
async def search(request: QueryRequest):
    global request_count
    query = request.query
    print(query)
    top_n = request.top_n
    return_score = request.return_score

    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            if (retriever_idx, query, top_n, return_score) in retriever_cache:
                results, scores = retriever_cache[(retriever_idx, query, top_n, return_score)]
            else:
                if return_score:
                    results, scores = cached_search(retriever_idx, query, top_n, return_score)
                    retriever_cache[(retriever_idx, query, top_n, return_score)] = (results, scores)
                else:
                    results = cached_search(retriever_idx, query, top_n, return_score)
                    retriever_cache[(retriever_idx, query, top_n, return_score)] = (results, None)

            request_count += 1
            if request_count % CACHE_SAVE_INTERVAL == 0:
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(retriever_cache, f)

            if return_score:
                return [Document(id=result['id'], contents=result['contents']) for result in results], scores
            else:
                return [Document(id=result['id'], contents=result['contents']) for result in results]
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/batch_search", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_search(request: BatchQueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            if return_score:
                results, scores = retriever_list[retriever_idx].batch_search(query, top_n, return_score)
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))], scores
            else:
                results = retriever_list[retriever_idx].batch_search(query, top_n, return_score)
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        finally:
            available_retrievers.append(retriever_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="./serving_config.yaml",
        help="path to serving config"
    )
    parser.add_argument(
        "--num_retriever", 
        type=int, 
        default=2,
        help="number of retriever to use, more retriever means more memory usage and faster retrieval speed"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=1243,
        help="port to use for the serving"
    )
    args = parser.parse_args()
    
    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

