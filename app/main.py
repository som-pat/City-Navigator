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

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount("/static", StaticFiles(directory="static"), name="static")





@app.get('/',response_class=HTMLResponse)
async def initial_map(request:Request):
    return templates.TemplateResponse("overlay.html",{
        "request":request
    })

@app.get('/momo')
async def et_tu_momo():
    return{'Hello World'}