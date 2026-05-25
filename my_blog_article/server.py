import sys
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError
from json import JSONDecodeError
from langserve import add_routes
from packages.rag_mongo.chain import InputModel, rag_mongo_chain

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(
        str(exc.errors()),
        status_code=400
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(
        exc.detail,
        status_code=exc.status_code
    )

add_routes(app, rag_mongo_chain, path="/rag_mongo")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API server. You can go to /docs for API documentation or /rag_mongo/playground for the playground."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

@app.get("/openapi.json", include_in_schema=False)
def openapi():
    return app.openapi()