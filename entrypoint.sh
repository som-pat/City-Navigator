#!/bin/bash
# Wait until PostgreSQL is ready

# while ! pg_isready -h db -p 5432 -U transitadmin > /dev/null 2> /dev/null; do
#   echo "$(date) - waiting for database to start"
#   sleep 2
# done

# python ETL_metro.py

# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload