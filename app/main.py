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
from collections import defaultdict
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from typing import List
from sqlalchemy import create_engine, text
import re


DATABASE_URL = "postgresql://transitpost:transitpost@postgres:5432/transitpost"
engine = create_engine(DATABASE_URL)
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

@app.get('/momo')
async def et_tu_momo():
    # conn = get_db_connection()
    # cur = conn.cursor()
    # cur.execute("Select * From search")
    # rows = cur.fetchall()
    # cur.close()
    # conn.close()

    # if not len(rows):
    #     sty = ['Empty']
    # else:
    #     sty = [row for row in rows]

   
    return {12345}


@app.get("/", response_class=HTMLResponse)
async def get_overlay(request: Request):


    return templates.TemplateResponse("overlay.html", {
        "request": request
    })


@app.get("/api/getStops")
async def get_stops(Type: str = Query(...), query: str = Query(...)):
    conn = engine.connect()
    try:
        sql1 = """
            SELECT multimodal
            FROM search
            WHERE type = :type AND multimodal ILIKE :query
            ORDER BY multimodal
            LIMIT 20;
        """
        sql2 = """
            SELECT multimodal
            FROM search
            WHERE multimodal ILIKE :query
            ORDER BY multimodal
            LIMIT 20;
        """

        search_query = f"%{query}%"
        if Type =='Bus' or Type == 'Metro':
            result = conn.execute(text(sql1), {"type": Type, "query": search_query})
            stops = [row[0] for row in result.fetchall()]
        elif Type == 'multimodal' or Type == 'walking':
            result = conn.execute(text(sql2), {"type": Type, "query": search_query})
            stops = [row[0] for row in result.fetchall()]

    except Exception as e:
        print(f"Error fetching stops: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        conn.close()
    return JSONResponse(content={"stops": stops})

@app.post('/searchRoute',response_class=JSONResponse)
async def searchRoute(request: Request, start_point: str = Form(...), 
                       end_point: str = Form(...),  transport_type:str = Form(...) ):
    
    start = (re.search(r'\((.*?)\)',str(start_point))).group(1) # type: ignore
    end = (re.search(r'\((.*?)\)', str(end_point))).group(1) # type: ignore
    transport_type = str(transport_type)
    print(start, end,transport_type)

    if transport_type == 'Metro':
        start = start.split('_')[0]
        end = end.split('_')[0]
        print(start, end,transport_type)
        graph = construct_metro_graph()        
        shortest_path = ShortestPath(graph, int(start), int(end))
        print(shortest_path)
        dist, time, route_time,stop_to_route = dist_route_time(shortest_path)
        path_coords = []
        route_name = []
        conn = get_db_connection()
        stopcur = conn.cursor()
        stop_time = []
        i=0
    
        for stop_id in shortest_path:
            stopcur.execute("SELECT stop_lat, stop_lon, stop_name FROM stops WHERE stop_id = %s", (stop_id,))
            stop = stopcur.fetchone()
            path_coords.append([stop[0], stop[1]]) # type: ignore
            route_name.append(stop[2]) # type: ignore        
            stop_time.append({'name':stop[2],'time':route_time[i]+' + 20s','route':stop_to_route[stop_id]})# type: ignore
            i+=1
    
    elif transport_type == 'Bus':
        start_stop = str(start).split('_')[1]
        end_stop = str(end).split('_')[1]

        graph = constructBus_graph()
        short_path = ShortestPath(graph, int(start_stop), int(end_stop))
        print(short_path)

        # conn = get_db_connection()
        # cur = conn.cursor()
        # stop_ls = []
        # for stop_id in short_path:
        #     cur.execute("SELECT stop_id,stop_lat, stop_lon, stop_name FROM buses_stops WHERE stop_id = %s", (stop_id,))
        #     stop_ls.append(cur.fetchone())
        # cur.close()
        # conn.close()

        # print(stop_ls)
        # route_id = len(route_store)
        # route_store[route_id] = stop_ls
    
    elif transport_type == 'multimodal':
        graph = construct_graph_multimodal()
        path = shortest_path_multimodal(graph, start, end)
        print(path)
        


    
    return 

def construct_metro_graph():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Fetch stops data
    cur.execute("SELECT stop_id FROM stops")
    stops = cur.fetchall()
    
    # Fetch stop_times and trips data to construct the graph
    cur.execute("""
        SELECT  st1.stop_id, st2.stop_id, st2.point_distance, t.route_id 
        FROM stop_times st1
        JOIN stop_times st2 ON st1.trip_id = st2.trip_id AND st1.stop_sequence + 1 = st2.stop_sequence
        JOIN trips t ON st1.trip_id = t.trip_id;
    """)
    stop_connections = cur.fetchall()
    cur.close()
    conn.close()
    
    graph = {stop[0]: [] for stop in stops}
    
    for connection in stop_connections:        
        graph[connection[0]].append((connection[1], connection[2]))
        graph[connection[1]].append((connection[0], connection[2]))  # undirected graph    
    
    return graph

def ShortestPath(graph, start, end):
    queue = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_nodes = {node: None for node in graph}
    
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))
                
                
    path, current_node = [], end
    while previous_nodes[current_node] is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    if path:
        path.append(start)
        
    return path[::-1]

def dist_route_time(shortest_path):
    conn = get_db_connection()
    cur = conn.cursor()    
    cur.execute("""
    SELECT stop_id, stop_sequence, point_distance, 
	EXTRACT(EPOCH FROM (individual_time::interval)), trips.route_id, individual_time, route.route_color
	FROM stop_times
	Join trips ON trips.trip_id = stop_times.trip_id
    Join route on route.route_id = trips.route_id    
    """)
    route_stop_time = cur.fetchall()
    
    # trip_change= defaultdict(list)
    stop_to_route = {}
    route_time = []
    dist = 0
    time = 0
    sp = []
    for i in shortest_path:
        cur.execute("SELECT stop_id, stop_sequence FROM stop_times WHERE stop_id = %s", (i,))
        st = cur.fetchone()
        sp.append([st[0],st[1]])# type: ignore
    now_time = datetime.datetime.now()
    now_time = now_time.strftime("%H:%M:%S")
    cur.close()
    conn.close()
    
    for m in range(len(sp)):
        for rst in route_stop_time:
            if sp[m][0] == rst[0] and sp[m][1] == rst[1]:

                stop_to_route[sp[m][0]] = rst[6]
                dist += rst[2]
                time += int(rst[3])
                route_time.append(time_adder([now_time,rst[5]]))
                now_time = route_time[-1]                
                break
    return  format(dist/1000, ".2f") ,format(time/60,".2f") , route_time  ,stop_to_route

def time_adder(time_list):
    tsum = datetime.timedelta()
    for i in time_list:
        (h,m,s) = i.split(':')
        t = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        tsum += t
    return str(tsum) 

def constructBus_graph():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Fetch stops data
    cur.execute("SELECT stop_id FROM buses_stops")
    stops = cur.fetchall()
    
    # Fetch stop_times and trips data to construct the graph
    cur.execute("""
    Select bc.stop_id , bc2.stop_id ,bc2.estimated_distance
    From buses_congo bc 
    Join buses_congo bc2 ON bc.trip_id = bc2.trip_id 
                AND bc.stop_sequence+1 = bc2.stop_sequence
    Order by bc.trip_id
    """)
    stop_connections = cur.fetchall()
    cur.close()
    conn.close()
    graph = {stop[0]: [] for stop in stops}
    
    for connection in stop_connections:
                # 3757:[3758,37]
                # 3758:[3757,37]
        val = float(format((connection[2]*1000), '.2f'))
        graph[connection[0]].append((connection[1], val))
        graph[connection[1]].append((connection[0], val))  # undirected graph
    
    return graph


def construct_graph_multimodal():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(""" 
    Select dep_stop,arr_stop,distance_km,time_secs,multimodal_type
    From combi
    UNION 
    Select stop1,stop2,distance_km, time,multimodal_type
    From walking
""")

    stop_connection = cur.fetchall()
    cur.close()
    conn.close()

    graph = defaultdict(list)

    for connection in stop_connection:
        graph[connection[0]].append([connection[1],connection[2],connection[4]])
        graph[connection[1]].append([connection[0],connection[2],connection[4]])

    return graph 

def shortest_path_multimodal(graph, start, end):
    queue = [(0, start, None)]  # (current_distance, current_node, current_mode)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_nodes = {node: None for node in graph}
    previous_modes = {node: None for node in graph}
    
    while queue:
        current_distance, current_node, current_mode = heapq.heappop(queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight, mode in graph[current_node]:
            if mode != current_mode:
                # Add some cost if there's a mode change
                weight += 5  # Example: penalty for changing modes
            
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                previous_modes[neighbor] = mode
                heapq.heappush(queue, (distance, neighbor, mode))
    
    path, current_node = [], end
    while previous_nodes[current_node] is not None:
        path.append((current_node, previous_modes[current_node]))
        current_node = previous_nodes[current_node]
    if path:
        path.append((start, previous_modes[start]))
    print(path[::-1])
    return path[::-1]