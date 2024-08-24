import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import os

db_params = {
    'host': os.getenv('DB_HOST', 'postgdb'),  
    'database': os.getenv('DB_NAME', 'gtfs_del'),
    'user': os.getenv('DB_USER', 'transitadmin'),
    'password': os.getenv('DB_PASS', 'gtfsuser0000')
}

# Create a connection to the PostgreSQL server
conn = psycopg2.connect(
    host=db_params['host'],
    database=db_params['database'],
    user=db_params['user'],
    password=db_params['password']
)
# Create a cursor object
cur = conn.cursor()

# Set automatic commit to be true, so that each action is committed without having to call conn.committ() after each command
conn.set_session(autocommit=True)

# Commit the changes and close the connection to the default database
conn.commit()
cur.close()
conn.close()



# db_params['database'] = 'gtfs_del'
engine = create_engine(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}/{db_params["database"]}')

# Define the file paths for your txt files

csv_files = {
    'route': 'Dataset/routes4.txt',
    'shape': 'Dataset/shapes.txt',
    'stop_times':'Dataset/stop_time3.txt',
    'stops': 'Dataset/stops.txt',
    'trips': 'Dataset/trips.txt'
}



# Loop through the CSV files and import them into PostgreSQL
for table_name, file_path in csv_files.items():
    df = pd.read_csv(file_path)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Data from {file_path} has been uploaded to the {table_name} table.")