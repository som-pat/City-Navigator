from fastapi import FastAPI, status, Request, Form, HTTPException, Depends
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
import random


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


@app.get("/api/getCoordinates")
async def get_coordinates(start: str, end: str):
    # Fetch coordinates from your database based on start and end stop names
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT stop_lat, stop_lon FROM search WHERE multimodal = %s", (start,))
    start_coords = cur.fetchone()
    
    cur.execute("SELECT stop_lat, stop_lon FROM search WHERE multimodal = %s", (end,))
    end_coords = cur.fetchone()

    cur.close()
    conn.close()
    

    if not start_coords or not end_coords:
        return JSONResponse(content={"error": "Coordinates not found"}, status_code=404)

    return JSONResponse(content={
        "start": {"lat": start_coords[0], "lng": start_coords[1]},
        "end": {"lat": end_coords[0], "lng": end_coords[1]}
    })



@app.post("/api/getRouteSummary",response_class=JSONResponse)
async def searchRoute(request: Request, start_point: str = Form(...), 
                       end_point: str = Form(...),  transport_type:str = Form(...) ):
    
    start = (re.search(r'\((.*?)\)',str(start_point))).group(1) # type: ignore
    end = (re.search(r'\((.*?)\)', str(end_point))).group(1) # type: ignore
    route_summary = []
    

    if transport_type == 'Metro':
        
        start = start.split('_')[0]
        end = end.split('_')[0]
        graph = construct_metro_graph()
        whole = ShortestPath(graph,int(start), int(end),'Metro')
        
        for i in whole['stop_details'] :
            route_summary.append({'name':i['stop_name'], 
                                  'time':i['time_instance'], 
                                  'route_fact':i['route_fact']})    


       

        
    
    elif transport_type == 'Bus':
        start_stop = str(start).split('_')[1]
        end_stop = str(end).split('_')[1]

        graph = constructBus_graph()
        whole = ShortestPath(graph, int(start_stop), int(end_stop),'Bus')
        for i in whole['stop_details'] :
            route_summary.append({'name':i['stop_name'], 
                                'time':i['time_instance'], 
                                'route_fact':i['route_fact']}) 


         
    
    elif transport_type == 'multimodal':
        graph = construct_graph_multimodal()
        whole = shortest_path_multimodal(graph, start, end)
        print(whole)
        
        # for i in whole['stop_details'] :
        #     route_summary.append({'name':i['stop_name'], 
        #                           'time':i['time_instance'], 
        #                           'route_fact':i['route_fact']})



    return JSONResponse(content={"path":whole['path_coords'] ,   # map routing 
                                 "route_summary":route_summary,  # summary details
                                 "dist":whole['total_distance'],#distance:summary 
                                 "time":whole['total_time']})    # time:summary  





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

def constructBus_graph():
    conn = get_db_connection()
    cur = conn.cursor()
    
   
    cur.execute("SELECT stop_id FROM buses_stops")
    stops = cur.fetchall()
    
    
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

def ShortestPath(graph, start, end, type):
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


    whole = route_stop_details(path[::-1], type)    

    return whole

def time_adder(tt, t):
    (h,m,s) = t.split(':')
    t = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
    platform_delay = datetime.timedelta(seconds= random.randint(5,8))
    tt += t + platform_delay
    return tt


def route_stop_details(path,type):
    conn = get_db_connection()
    stop_details = []
    path_coords = []
    total_time = 0
    total_distance = 0
    start_time = 0
    end_time = 0
    whole ={}

    print(1,path)
    try:

        total_time = datetime.datetime.now()
        (h,m,s) = (total_time.strftime("%H:%M:%S")).split(':')
        total_time = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        
        start_time = total_time

        if type == 'Metro': 

            metro_cur = conn.cursor()
            for i in path:                
                
                metro_cur.execute("""Select individual_time,point_distance,stop_name,
                                  route_color,stop_lat,stop_lon From metro_result where stop_id =%s""",(i,))
                
                st = metro_cur.fetchone()
                total_distance+=st[1]/1000
                total_time = time_adder(total_time,st[0])

                path_coords.append({
                    'stop_name': st[2],
                    'geopoints': [st[4],st[5]],
                    'stop_lat': st[4],
                    'stop_lon':st[5]
                })
                
                stop_details.append({
                    'stop_id': i,
                    'stop_name': st[2],
                    'time_instance': str(total_time).split('.')[0],
                    'distance': st[1],
                    'route_fact': st[3]
                })
            
            end_time = str(total_time-start_time).split('.')[0]
            metro_cur.close()
            conn.close()

        elif type =='Bus':
            
            bus_cur = conn.cursor()
            for i in path:                
                
                bus_cur.execute("""Select individual_time,estimated_distance,stop_name, 
                                stop_lat,stop_lon, route_long_name From bus_result where stop_id =%s""",(i,))
                
                st = bus_cur.fetchone()
                total_distance += st[1]
            
                total_time = time_adder(total_time,st[0])

                path_coords.append({
                    'stop_name': st[2],
                    'geopoints': [st[3],st[4]],
                    'stop_lat': st[3],
                    'stop_lon':st[4]
                    
                })

                stop_details.append({
                    'stop_id': i,
                    'stop_name': st[2],
                    'time_instance': str(total_time).split('.')[0],
                    'distance': st[1],
                    'route_fact': st[5]
                })

            end_time = str(total_time-start_time).split('.')[0]
            bus_cur.close()
            conn.close()

    except Exception as e:
        print(f"Error passing stop_details: {str(e)}")
        
    finally:
        whole['path_coords'] = path_coords
        whole['stop_details'] =  stop_details
        whole['total_time'] = end_time
        whole['total_distance'] = total_distance
    
    return whole




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
                # Add cost on mode change
                weight += 5  # penalty for changing modes
            
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
    return path[::-1]