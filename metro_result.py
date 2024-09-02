#TODO requirements_metro - cumulative time of trip(main), intra-time(db), dep_stop_name(db)
#                          stop_id and stop_sequence(db), route_color(db), intra-distance(db)
#                          cumulative_trip_distance(main), stop_lat & stop_lon(db)
# TODO requirements_bus  - cumulative time of trip(main), intra-time(db), dep_stop_name(db)
#                          stop_id and stop_sequence(db), intra-distance(db)
#                          cumulative_trip_distance(main), stop_lat & stop_lon(db)

import pandas as pd

metrosc = pd.read_csv('Dataset/results/metro_stop_connection.csv') 
routes  = pd.read_csv('Dataset/metro/routes4.txt')
trips = pd.read_csv('Dataset/metro/trips.txt')
s_time = pd.read_csv('Dataset/metro/stop_time3.txt')
stops = pd.read_csv('Dataset/metro/stops.txt')

# s_time = s_time.drop(['stop_headsign','pickup_type', 'drop_off_type', 
#                       'shape_dist_traveled', 'timepoint','continuous_pickup', 
#                       'continuous_drop_off'], axis=1, errors='ignore') 

# trc = s_time.set_index('stop_id').join(stops.set_index('stop_id'), how='inner')
feat_tr = ['route_id','trip_id']
feat_rc = ['route_id','route_color']
trc = trips[feat_tr].set_index('route_id').join(routes[feat_rc].set_index('route_id'), how='inner')
trc1 = pd.merge(s_time, stops, on='stop_id',how='inner' )
trc1= trc1.drop(['stop_headsign','pickup_type', 'drop_off_type', 
                      'shape_dist_traveled', 'timepoint','continuous_pickup', 
                      'continuous_drop_off','stop_code','stop_desc'], axis=1, errors='ignore')
trc2 =  pd.merge(trc, trc1, on='trip_id',how='inner' )

# Display the resulting dataframe
print(trc1)
print(trc1.head().to_string())

print(trc2)
print(trc2.info())
print(trc2.route_color.value_counts())

print(trc2.groupby('route_color')['trip_id'].nunique())
# metrosc = pd.merge(metrosc,trc)

# print(metrosc)
# print(metrosc.info())
# # print(trc.route_color.value_counts())
trc2.to_csv('Dataset/results/res22.csv',index=False)
# print(metrosc.columns)
# print(routes.columns)
# print(trips.columns)