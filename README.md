# City_Navigator

## About

#### This project is a dockerized web application that displays routes from one point to another based on metro, bus, walking and multimodial(mix of all types) for a city using FastAPI, Leaflet.js, Python, PostgreSQL, Docker and machine learning algorithms. Users can input start and end points for their journey to find an efficient and shortest route. Factors like less walking and faster journey routes can be set and the algorithm will include those and map the route dynamically.

## Features

- Finds the fastest travellable route on the basis of metro, bus, walking or a mix of all.
- Routes take into account the weather possibilites, traffic conditions and selects travel means for the fastest ETA. 


Project Structure
``` bash
project_root/
│
├── Dataset/
│   ├── Metro_zipfile
│   ├── DMTC_buses_zipfile
│
├── app/
│   ├── main.py
│   
├──Static/
│  ├──styles.css
│
├──Templates/
│  ├──overlay.html
│ 
├── Dockerfile
├── compose.yml
├── requirements.txt
└── ETL_combi.py
```

For running the project follow the below instructions 
```
- Clone the repo : https://github.com/som-pat/Metro.git 
- Start virtual environment: .venv\Scripts\Activate
- Download the datasets from the links below
- Populate the database : docker-compose start pgadmin && python ETL_combi.py
- Start the whole app : docker-compose up
- Application will be running on localhost:8000
```
#### This project follows the guidelines from the General Transit Feed Specification (GTFS) for its working and the data used is static, ETA shown is based on historical data.
##### GTFS link 
- [https://gtfs.org/]
- [https://developers.google.com/transit/gtfs]
##### Static Metro GTFS dataset
- [https://otd.delhi.gov.in/data/staticDMRC/]
##### Static Bus GTFS dataset
- [https://otd.delhi.gov.in/data/static/]
##### Weather data
- [https://www.kaggle.com/datasets/mahirkukreja/delhi-weather-data]
