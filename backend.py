from fastapi import FastAPI, status, Request, Form, HTTPException
from typing import List
import uvicorn
import psycopg2
import heapq
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse,JSONResponse,RedirectResponse
import os
import datetime

templates = Jinja2Templates(directory="templates")
app = FastAPI()

@app.get('/')
def read_root():
    return {'Hello World'}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)