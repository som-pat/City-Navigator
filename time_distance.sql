SELECT st1.trip_id,st1.stop_id,st2.stop_id, st2.point_distance,st1.departure_time,st2.arrival_time,
	t.route_id,EXTRACT(EPOCH FROM (st2.individual_time::interval)) as travel_time
        FROM stop_times st1
        JOIN stop_times st2 
            ON st1.trip_id = st2.trip_id 
            AND st1.stop_sequence + 1 = st2.stop_sequence
		JOIN trips t ON st1.trip_id = t.trip_id
Order by st1.trip_id,st1.stop_id, st2.stop_id;

SELECT stop_id, stop_sequence, shape_dist_traveled, point_distance, 
	EXTRACT(EPOCH FROM (individual_time::interval)) as travel_time,trips.route_id
	FROM stop_times
	Join trips ON trips.trip_id = stop_times.trip_id

SELECT st1.trip_id,st1.stop_id, st2.stop_id,st2.arrival_time::time as t2, st1.departure_time::time as t1, st2.point_distance, 
	t.route_id,EXTRACT(EPOCH FROM (st2.arrival_time::time - st1.departure_time::time)) as travel_time, st2.individual_time
        FROM stop_times st1
        JOIN stop_times st2 
            ON st1.trip_id = st2.trip_id 
            AND st1.stop_sequence + 1 = st2.stop_sequence
		JOIN trips t ON st1.trip_id = t.trip_id
Order by st1.trip_id,st1.stop_id, st2.stop_id;



SELECT  st1.stop_id, st2.stop_id, st2.point_distance, t.route_id 
        FROM stop_times st1
        JOIN stop_times st2 ON st1.trip_id = st2.trip_id AND st1.stop_sequence + 1 = st2.stop_sequence
        JOIN trips t ON st1.trip_id = t.trip_id;