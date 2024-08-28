import pandas as pd
import psycopg2
from sqlalchemy import create_engine


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

cd