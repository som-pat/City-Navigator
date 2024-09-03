import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from zipfile import ZipFile 
import geopy.distance as gd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.metrics import silhouette_score

# with ZipFile("Dataset/buses/GTFS_bus.zip", 'r') as zObject: 
# 	zObject.extractall( 
# 		path="Dataset/buses") 


DATABASE_URL = "postgresql://transitpost:transitpost@localhost:5432/transitpost"
engine = create_engine(DATABASE_URL)

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# conn = get_db_connection()
# cur = conn.cursor()
# cur.execute("Select * From student")
# rows = cur.fetchall()

# if not len(rows):
#     print("Empty")
# else:
#     for row in rows:
#         print(row)

# cur.close()

# with ZipFile("Dataset/GTFS_bus.zip", 'r') as zObject: 
# 	zObject.extractall( 
# 		path="Dataset/buses") 

# with ZipFile("Dataset/DMRC_GTFS.zip", 'r') as zObject: 
# 	zObject.extractall( 
# 		path="Dataset/metro")

# #Metro
# # Route.txt Conversion
# routes = pd.read_csv('Dataset/metro/routes.txt')
# def seperator(route_long_name):
#     parts = route_long_name.split('_')
#     if 'RAPID' in parts:
#         parts[0] ='PURPLE'
#     if 'ORANGE/AIRPORT' in parts:
#         parts[0] ='ORANGE'
#     if 'GRAY' in parts:
#         parts[0] ='#0C0C0C'
#     if 'YELLOW' in parts:
#         parts[0] = '#750E21'
#     if 'MAGENTA' in parts:
#         parts[0] = '#720455'

#     color = parts[0] if len(parts)>1 else None
#     if 'to' in parts[-1]:
#         rt = parts[-1].split(' to ')
#         start_point = rt[0]
#         end_point = rt[1]
#     else:
#         start_point=end_point=None
    
#     return pd.Series([color,start_point,end_point])

# routes[['route_color','start_point','end_point']] = routes['route_long_name'].apply(seperator)
# routes = routes.sort_values(by=['route_color'])

# print('1.routes')
# print(routes.head().to_string())

# # empty the text file if previously used to prevent duplication
# # routes.to_csv('Dataset/routes4.txt', header=True, index=None, sep=',', mode='a') # type: ignore

# # Stop_times.txt conversion
# stop_time = pd.read_csv('Dataset/metro/stop_times.txt')

def normalize_time(time_str):    
    h, m, s = map(int, time_str.split(':'))    
    if h >= 24:
        h = h % 24        
    return f"{h:02}:{m:02}:{s:02}"


# stop_time['arrival_time'] = stop_time['arrival_time'].apply(normalize_time)
# stop_time['departure_time'] = stop_time['departure_time'].apply(normalize_time)
# # time.to_csv('Dataset/stop_time2.txt', header=True, index=None, sep=',', mode='a') # type: ignore
# print('2.stop_time')
# print(stop_time.head().to_string())

# # Stops for 20 secs at each station
# # time = next stop arrival_id - previous stop departure_id 
# # distance = next stop_id -  previous stop_id

# def process_trip(trip_df):
    
#     # Sort by stop_sequence
#     trip_df = trip_df.sort_values(by='stop_sequence')
#     trip_df['point_distance'] = trip_df['shape_dist_traveled'].diff().fillna(0)
#     # print(trip_df[['trip_id','stop_id','stop_sequence','point_distance']])
#     trip_df['arrival_time'] = pd.to_timedelta(trip_df['arrival_time'])
#     trip_df['departure_time'] = pd.to_timedelta(trip_df['departure_time'])
#     trip_df['individual_time'] = (trip_df['arrival_time'] - trip_df['departure_time'].shift()).fillna(pd.Timedelta(seconds=0))

#     trip_df['arrival_time'] = trip_df['arrival_time'].apply(lambda x: str(x).replace('0 days ', ''))
#     trip_df['departure_time'] = trip_df['departure_time'].apply(lambda x: str(x).replace('0 days ', ''))    
#     trip_df['individual_time'] = trip_df['individual_time'].apply(lambda x: str(x).replace('0 days ', ''))
#     trip_df['individual_time'] = trip_df['individual_time'].apply(lambda x: str(x).replace('-1 days ', ''))
#     return trip_df

# stop_time = stop_time.groupby('trip_id').apply(process_trip).reset_index(drop=True)
# print('3.stop_time')
# print(stop_time.head().to_string())

# # stop_time.to_csv('Dataset/stop_time3.txt', header=True, index=None, sep=',', mode='a') # type: ignore

# #Bus
bus_stoptime = pd.read_csv('Dataset/buses/stop_times.txt')

def cal_dist_latlon(row):
    if (row['stop_lat_lag'] == 0) and (row['stop_lon_lag'] == 0):
        return 0
    return gd.geodesic((row['stop_lat_lag'], row['stop_lon_lag']), (row['stop_lat'], row['stop_lon'])).km

def stop_aggregation():
    stop_df = pd.read_csv('Dataset/buses/stops.txt')
    stop_df['stop_code_id'] = stop_df['stop_code'] + '_' + stop_df['stop_id'].astype(str)
    stop_df.to_csv('Dataset/buses/stop2.txt', header=True, index=None, sep=',', mode='a') # type: ignore

bus_stoptime = pd.read_csv('Dataset/buses/stop_times.txt')
stop_aggregation()

bus_stoptime ['arrival_time'] = bus_stoptime ['arrival_time'].apply(normalize_time)
bus_stoptime ['departure_time'] = bus_stoptime ['departure_time'].apply(normalize_time)

print('4.bus_stop_time')
print(bus_stoptime.head().to_string())

df2 = pd.read_csv('Dataset/buses/trips.txt')
bus_stoptime = pd.merge(bus_stoptime,df2, on='trip_id')
df4 = pd.read_csv('Dataset/buses/routes.txt')
bus_stoptime = pd.merge(bus_stoptime,df4, on ='route_id')
df5 =pd.read_csv('Dataset/buses/stops.txt')
bus_stoptime = pd.merge(bus_stoptime,df5,on='stop_id')
print('4.2 bus_stop_time')
print(bus_stoptime.columns)
print(bus_stoptime.head().to_string())
bus_stoptime = bus_stoptime.drop(['service_id','shape_id', 'agency_id',
                                  'route_short_name','zone_id','route_type'],axis=1)

print('5.bus_stop_time')
print(bus_stoptime.head().to_string())

def process_bus_trip(trip_df):    
    # Sort by stop_sequence
    trip_df = trip_df.sort_values(by='stop_sequence')    

    trip_df['arrival_time'] = pd.to_timedelta(trip_df['arrival_time'])
    trip_df['departure_time'] = pd.to_timedelta(trip_df['departure_time'])
    trip_df['individual_time'] = (trip_df['arrival_time'] - trip_df['departure_time'].shift()).fillna(pd.Timedelta(seconds=0))

    trip_df['arrival_time'] = trip_df['arrival_time'].apply(lambda x: str(x).replace('0 days ', ''))
    trip_df['departure_time'] = trip_df['departure_time'].apply(lambda x: str(x).replace('0 days ', ''))    
    trip_df['individual_time'] = trip_df['individual_time'].apply(lambda x: str(x).replace('0 days ', ''))
    trip_df['individual_time'] = trip_df['individual_time'].apply(lambda x: str(x).replace('-1 days ', ''))
    return trip_df


#arrival and departure time format, distance
bus_stoptime = bus_stoptime.groupby('trip_id').apply(process_bus_trip).reset_index(drop=True)
# fin_df.to_csv('Dataset/buses/stop_times3.txt', header=True, index=None, sep=',', mode='a') # type: ignore

#individual time to secs, hour of day inclusion
bus_stoptime['convtime_secs'] = pd.to_timedelta(bus_stoptime['individual_time']).dt.seconds
bus_stoptime['day_hour'] = pd.to_datetime(bus_stoptime['arrival_time'], format='%H:%M:%S').dt.hour

# Fill value is taken as 0 so that NaN does not affect calculations
# The lags for every trips 1st coordinates will be 0 as trips start from there
bus_stoptime['stop_lat_lag'] = bus_stoptime.groupby('trip_id')['stop_lat'].shift(1,fill_value=0)
bus_stoptime['stop_lon_lag'] = bus_stoptime.groupby('trip_id')['stop_lon'].shift(1, fill_value=0)
bus_stoptime['estimated_distance'] = bus_stoptime.apply(cal_dist_latlon, axis=1)

bus_stoptime['cumulative_distance'] = bus_stoptime.groupby('trip_id')['estimated_distance'].cumsum()
bus_stoptime['cumulative_time'] = bus_stoptime.groupby('trip_id')['convtime_secs'].cumsum()

bus_stoptime['route_id_encoded'] = bus_stoptime['route_id'].astype('category').cat.codes
bus_stoptime['stop_name_encoded'] = bus_stoptime['stop_name'].astype('category').cat.codes

print('6.bus_stop_time')
print(bus_stoptime.head().to_string())
print(bus_stoptime.shape)

bus_stoptime.to_csv('Dataset/buses/final_stop_times.csv',index=False) # type: ignore


# #Multimodal

# bus_stop = pd.read_csv('Dataset/buses/stop2.txt')
# metro_stop = pd.read_csv('Dataset/metro/stops.txt')

# def generate_stop_code(row):
#     word = row['stop_name'].split(' ')
#     if len(word) > 1:
#         return f"{row['stop_id']}_{word[0][0]}{word[1][0]}"
#     else:
#         return f"{row['stop_id']}_{word[0][:2]}"

# metro_stop['stop_code_id'] = metro_stop.apply(generate_stop_code, axis=1)
# metro_stop = metro_stop.drop(['stop_id','stop_desc','stop_code'],axis = 1)
# bus_stop = bus_stop.drop(['stop_id', 'zone_id','stop_code'],axis=1)
# bus_stop['tratype']  = 'Bus'

# df = metro_stop
# df['tratype'] = 'Metro'
# df = pd.concat([df,bus_stop],ignore_index = True)

# print('7.Multimodal_stop_time')
# print(df.head().to_string())
# print(df.shape)

# #Region clustering
# def kmeans_inertia(num_clusters, coords):
#     inertia = []
#     sil_score =[]
#     for k in num_clusters:
#         kms = KMeans(n_clusters=k, random_state=42).fit(coords)
#         inertia.append(kms.inertia_)
#         sil_score.append(silhouette_score(coords,kms.labels_))
    
#     print(inertia)
#     plt.plot(num_clusters,inertia)
#     plt.xlabel("Number of clusters")
#     plt.ylabel("inertia")
#     plt.show(block=True)
    
#     print(sil_score)
#     plt.plot(num_clusters,sil_score)
#     plt.xlabel("Number of clusters")
#     plt.ylabel("Sillhouette Score")
#     plt.show(block=True)

# num_clusters = [i for i in range(2,25)]
# coords = df[['stop_lat', 'stop_lon']].values
# kmeans_inertia(num_clusters, coords)
# kms = KMeans(n_clusters=10, random_state=42).fit(coords)
# plt.scatter(coords[:, 0], coords[:, 1], c=kms.labels_, cmap='Paired')
# plt.title("K-Means Clustering")
# plt.show(block=True)

# df['cluster'] = kms.labels_
# dsc = df['cluster'].value_counts().to_dict()
# print(dsc)

# distance_within_clusters = []
# for cluster_id in df['cluster'].unique():
#     cluster_stops = df[df['cluster'] == cluster_id]

#     #calculate distance and time 
#     for i, stop1 in cluster_stops.iterrows():
#         for j, stop2 in cluster_stops.iterrows():
#             if i<j and stop1['tratype'] != stop2['tratype']:
#                 point1 = (stop1['stop_lat'],stop1['stop_lon'])
#                 point2 = (stop2['stop_lat'],stop2['stop_lon'])
#                 distance = geodesic(point1,point2).km
#                 walking_speed = 5 #kmph
#                 time = distance/walking_speed * 60 * 60 #convert to secs

#                 #store
#                 distance_within_clusters.append({
#                     'cluster':str(cluster_id),
#                     'stop1':stop1['stop_code_id'], 
#                     'stop_type1':stop1['tratype'],
#                     'stop2':stop2['stop_code_id'], 
#                     'stop_type2':stop2['tratype'],
#                     'distance_km': distance,
#                     'time':time,
#                     'multimodal_type':'walking'
#                 })


# mbs_dwc = []
# for cluster_id in df['cluster'].unique():
#     cluster_stops = df[df['cluster'] == cluster_id]

#     #calculate distance and time 
#     for i, stop1 in cluster_stops.iterrows():
#         for j, stop2 in cluster_stops.iterrows():
#             if i<j and stop1['tratype'] == stop2['tratype']:
#                 point1 = (stop1['stop_lat'],stop1['stop_lon'])
#                 point2 = (stop2['stop_lat'],stop2['stop_lon'])
#                 distance = geodesic(point1,point2).km
#                 walking_speed = 5 #kmph
#                 time = distance/walking_speed * 60 * 60 #convert to secs
#                 if distance <= 1.5:
#                 #store
#                     mbs_dwc.append({
#                         'cluster':str(cluster_id),
#                         'stop1':stop1['stop_code_id'], 
#                         'stop_type1':stop1['tratype'],
#                         'stop2':stop2['stop_code_id'], 
#                         'stop_type2':stop2['tratype'],
#                         'distance_km': distance,
#                         'time':time,
#                         'multimodal_type':'walking'
#                     })


# dwc_df = pd.DataFrame(distance_within_clusters)
# mbs_dwc_df = pd.DataFrame(mbs_dwc)
# print('8.Multimodal_stop_time')
# print(df.head().to_string())
# print(dwc_df.head(10).to_string())
# print(dwc_df.shape)
# print(mbs_dwc_df.head(10).to_string())
# print(mbs_dwc_df.shape)

# dwc_df.to_csv('Dataset/results/dwc2.csv',index=False)
# mbs_dwc_df.to_csv('Dataset/results/mbs_dwc2.csv',index=False)

# metro_stop = df[df['tratype'] == 'Metro']
# bus_stop = df[df['tratype'] == 'Bus']

# def find_nearest_bus_stop(stop1, stop2, max_distance=0.5):
#     nearest_stop = None    
#     for _, st in stop2.iterrows():
#         distance = geodesic((stop1['stop_lat'], stop1['stop_lon']),
#                             (st['stop_lat'], st['stop_lon'])).km
#         if distance <= max_distance:
#             nearest_stop = st['stop_code_id']
#             break
#     return nearest_stop

# def find_nearest_metro_stop(bus_stop, metro_stop,max_distance=0.5):
#     nearest_metro_stop =None
#     for _, metro_stop in metro_stop.iterrows():
#         distance = geodesic((bus_stop['stop_lat'], bus_stop['stop_lon']),
#                             (metro_stop['stop_lat'],metro_stop['stop_lon'])).km
#         if distance <= max_distance:
#             nearest_metro_stop = metro_stop['stop_code_id']
#             break
#     return nearest_metro_stop


# metro_stop.loc[:, 'nearest_bus'] = metro_stop.apply(lambda x:find_nearest_bus_stop(x,bus_stop), axis=1)
# metro_to_bus_gap = metro_stop[metro_stop['nearest_bus'].isna()]

# bus_stop.loc[:, 'nearest_metro'] = bus_stop.apply(lambda x:find_nearest_metro_stop(x, metro_stop),axis=1)
# bus_to_metro_gap = bus_stop[bus_stop['nearest_metro'].isna()]


# print('9.Multimodal_stop_time')
# print(metro_stop.head(10).to_string())
# print(metro_to_bus_gap.head(10).to_string())

# print(bus_stop.shape)
# print(bus_stop.head(10).to_string())
# print(bus_to_metro_gap.head(10).to_string())
# print(bus_to_metro_gap.shape)

# metro_stop.to_csv('Dataset/results/metro_bus_near.txt', header=True, index=None, sep=',', mode='a') # type: ignore
# metro_to_bus_gap.to_csv('Dataset/results/metro_bus_NotNear.txt', header=True, index=None, sep=',', mode='a') # type: ignore

# bus_stop.to_csv('Dataset/results/bus_metro_near.txt', header=True, index=None, sep=',', mode='a') # type: ignore
# bus_to_metro_gap.to_csv('Dataset/results/bus_metro_NotNear.txt', header=True, index=None, sep=',', mode='a') # type: ignore


# #comment from here to the csv_files , execute csv_files first 
# # then execute csvsaver to generate the two csv files 
# intra_cluster = pd.read_csv('Dataset/results/dwc2.csv')
# metro_dist = pd.read_csv('Dataset/results/metro_stop_connection.csv')
# bus_dist = pd.read_csv('Dataset/results/bus_stop_connection.csv')



# def generate_metstop_code(row):
#     if 'dep_stop' not in metro_dist.columns or len(metro_dist['dep_stop'])!= metro_dist.shape[0]:
#         word = row['dep_stop_name'].split(' ')
#         if len(word) > 1:
#             return f"{row['dep_stop_id']}_{word[0][0]}{word[1][0]}"
#         else:
#             return f"{row['dep_stop_id']}_{word[0][:2]}"
#     if 'arr_stop' not in metro_dist.columns or len(metro_dist['arr_stop'])!= metro_dist.shape[0]:
#         word = row['arr_stop_name'].split(' ')
#         if len(word) > 1:
#             return f"{row['arr_stop_id']}_{word[0][0]}{word[1][0]}"
#         else:
#             return f"{row['arr_stop_id']}_{word[0][:2]}"

# def generate_bustop_code(row):
#     if 'dep_stop' not in bus_dist.columns or len(bus_dist['dep_stop'])!= bus_dist.shape[0]:
#         return f"{row['dep_stop_code']}_{row['dep_stop_id']}"
        
#     if 'arr_stop' not in bus_dist.columns or len(bus_dist['arr_stop'])!= bus_dist.shape[0]:
#         return f"{row['arr_stop_code']}_{row['arr_stop_id']}"




# metro_dist['dep_stop'] = metro_dist.apply(generate_metstop_code, axis=1) # type: ignore
# metro_dist['arr_stop'] = metro_dist.apply(generate_metstop_code, axis=1) # type: ignore

# bus_dist['dep_stop'] = bus_dist.apply(generate_bustop_code, axis=1) # type: ignore
# bus_dist['arr_stop'] = bus_dist.apply(generate_bustop_code, axis=1) # type: ignore

# metro_dist['distance_km']  = metro_dist['distance_m'].apply(lambda x : x/1000) #km 

# metro_dist['multimodal_type'] = 'Metro'
# bus_dist['multimodal_type'] = 'Bus'

 
# metro_dist = metro_dist.drop(['dep_stop_id', 'dep_stop_name', 'arr_stop_id','arr_stop_name','distance_m'], axis=1)
# bus_dist = bus_dist.drop(['dep_stop_code', 'dep_stop_id', 'arr_stop_code','arr_stop_id'], axis=1)
# intra_cluster = intra_cluster.drop(['time'],axis=1)

# df = metro_dist
# df = pd.concat([df, bus_dist], ignore_index=True)
# df['trip_id_cc'] = df['trip_id'].astype('category').cat.codes

# print('10.Multimodal_stop_time')
# print('Walking')
# print(intra_cluster.head(5).to_string())
# print(intra_cluster.shape)
# print('Metro')
# print(metro_dist.head(10).to_string())
# print(metro_dist.shape)
# print('Bus')
# print(bus_dist.head(10).to_string())
# print(bus_dist.shape)

# print('Combined')
# print(df.head(10).to_string())
# print(df.tail(10).to_string())

# df.to_csv('Dataset/results/combi_mb2.csv', index=False)

## namelatlon - txt file deleted

# metro_dist = pd.read_csv('Dataset/metro/stops.txt')
# bus_dist = pd.read_csv('Dataset/buses/stop2.txt')

# metro_dist['type'] = 'Metro'
# bus_dist['type'] = 'Bus'

# def generate_stop_code(row):
#     word = row['stop_name'].split(' ')
#     if len(word) > 1:
#         return f"{row['stop_id']}_{word[0][0]}{word[1][0]}"
#     else:
#         return f"{row['stop_id']}_{word[0][:2]}"

# def bus_append(row):
#     return f"{row['stop_name']}({row['stop_code_id']})"

# def metro_append(row):
#     return f"{row['stop_name']}({row['stop_code_id']})"


# metro_dist['stop_code_id'] = metro_dist.apply(generate_stop_code, axis=1)
# bus_dist['multimodal'] = bus_dist.apply(bus_append,axis=1 )
# metro_dist['multimodal'] = metro_dist.apply(metro_append, axis=1)

# metro_dist = metro_dist.drop(['stop_id','stop_code','stop_name','stop_desc',
#                               'stop_code_id' ],axis=1)

# bus_dist = bus_dist.drop(['stop_code', 'stop_id','stop_name'  ,'zone_id' ,'stop_code_id'], axis=1)                    
#                           


# df= pd.concat([metro_dist,bus_dist],ignore_index=True)

# print('11.Multimodal_stop_time')
# print(df.head().to_string())

# df.to_csv('Dataset/results/NameLatLon.csv',index=False)



# csv_files = {
#     'agency':'Dataset/metro/agency.txt',
#     'calendar':'Dataset/metro/calendar.txt',
#     'route': 'Dataset/metro/routes4.txt',
#     'shape': 'Dataset/metro/shapes.txt',
#     'stop_times':'Dataset/metro/stop_time3.txt',
#     'stops': 'Dataset/metro/stops.txt',
#     'trips': 'Dataset/metro/trips.txt',
#     'buses_agency':'Dataset/buses/agency.txt',
#     'buses_calendar':'Dataset/buses/calendar.txt',
#     'buses_route': 'Dataset/buses/routes.txt',
#     'buses_fare_attributes': 'Dataset/buses/fare_attributes.txt',
#     'buses_fare_rules':'Dataset/buses/fare_rules.txt',
#     'buses_stop_times':'Dataset/buses/stop_times2.txt',
#     'buses_stops': 'Dataset/buses/stops.txt',
#     'buses_trips': 'Dataset/buses/trips.txt',
#     'buses_st2': 'Dataset/buses/stop2.txt',
#     'buses_congo': 'Dataset/buses/s_times3.txt',
#     'combi':'Dataset/results/combi_mb.csv',
#     'walking':'Dataset/results/dwc2.csv',
#     'search':'Dataset/results/NameLatLon.csv'
# }


csv_files = {
    'metro_result': 'Dataset/results/res22.csv',
    'bus_result': 'Dataset/buses/final_stop_times.csv'
}

for table_name, file_path in csv_files.items():
    df = pd.read_csv(file_path)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Data from {file_path} has been uploaded to the {table_name} table.")