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

DATABASE_URL = "postgresql://transitadmin:gtfsuser0000@postgdb/gtfs_del"
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


def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn


@app.get('/',response_class=HTMLResponse)
async def initial_map(request:Request):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT stop_id,stop_name FROM buses_stops bs WHERE bs.stop_id IN ( SELECT MIN(stop_id) FROM buses_stops GROUP BY stop_name);")
    stops = cur.fetchall()
    cur.execute("""
    SELECT stop_name, string_agg(stop_code_id,',') AS stop_ids 
    FROM buses_st2
    GROUP BY stop_name
    """)
    bus_stops = cur.fetchall()
    cur.close()
    conn.close()
    stop_data = {stop[0]: stop[1] for stop in bus_stops}

    return templates.TemplateResponse("overlay.html", {
        "stop_data":stop_data,
        "stops":stops,
        "request": request})

@app.get('/momo')
async def et_tu_momo():
    return{'Hello World'}