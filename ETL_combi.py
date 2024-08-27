import pandas as pd
import psycopg2
from sqlalchemy import create_engine


conn = psycopg2.connect(
    database= "transitpost",
    user= "transitpost",
    password= "transitpost",
    host= "0.0.0.0",
    port = "5432"
)

cur = conn.cursor()
cur.execute("Select * From student")
rows = cur.fetchall()

if not len(rows):
    print("Empty")
else:
    for row in rows:
        print(row)

cur.close()