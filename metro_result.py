import pandas as pd

metrosc = pd.read_csv('Dataset/results/metro_stop_connection.csv') 
routes  = pd.read_csv('Dataset/metro/routes4.txt')
trips = pd.read_csv('Dataset/metro/trips.txt')
feat_tr = ['route_id','trip_id']
feat_rc = ['route_id','route_color']
trc = trips[feat_tr].set_index('route_id').join(routes[feat_rc].set_index('route_id'), how='inner')
print(trc)
print(trc.info())
print(trc.route_color.value_counts())

metrosc = pd.merge(metrosc,trc)

print(metrosc)
print(metrosc.info())
# print(trc.route_color.value_counts())
metrosc.to_csv('Dataset/results/metro_stop_routecon.csv',index=False)
# print(metrosc.columns)
# print(routes.columns)
# print(trips.columns)