#Execute 
import csv
import psycopg2

DATABASE_URL = "postgresql://transitadmin:gtfsuser0000@localhost/gtfs_del"
bus_output_file = 'Dataset/results/bus_stop_connection.csv'
metro_output_file = 'Dataset/results/metro_stop_connection.csv'

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

conn = get_db_connection()
cur = conn.cursor()


cur.execute("""
Select bc.trip_id,bc.stop_code, bc.stop_id ,bc2.stop_code, bc2.stop_id, bc2.estimated_distance,
	bc2.individual_time,bc.stop_sequence,bc2.stop_sequence
From buses_congo bc 
Join buses_congo bc2 ON bc.trip_id = bc2.trip_id AND bc.stop_sequence+1 = bc2.stop_sequence
Order by bc.trip_id
""")
bus_stop_connections = cur.fetchall()


cur.execute(""" 
    SELECT st1.trip_id,  st1.stop_id, s1.stop_name, st2.stop_id, s2.stop_name, st2.point_distance, EXTRACT(EPOCH FROM (st2.individual_time::interval)),
	st1.stop_sequence, st2.stop_sequence 
    FROM stop_times st1
    JOIN stop_times st2 ON st1.trip_id = st2.trip_id AND st1.stop_sequence + 1 = st2.stop_sequence
    JOIN trips t ON st1.trip_id = t.trip_id
	JOIN stops s1 ON st1.stop_id = s1.stop_id 
	JOIN stops s2 ON st2.stop_id = s2.stop_id 
Order by st1.trip_id

""")
metro_stop_connections = cur.fetchall() 

cur.close()
conn.close()



with open(bus_output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['trip_id','dep_stop_code','dep_stop_id','arr_stop_code'
                     ,'arr_stop_id','distance_km','time_secs','dep_stop_sequence','arr_stop_sequence'])
    writer.writerows(bus_stop_connections)

print(f'Bus Data Successfully saved to output file')

with open(metro_output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['trip_id','dep_stop_id','dep_stop_name','arr_stop_id','arr_stop_name',
                     'distance_m','time_secs','dep_stop_sequence','arr_stop_sequence'])
    writer.writerows(metro_stop_connections)

print(f'Metro Data Successfully saved to output file')