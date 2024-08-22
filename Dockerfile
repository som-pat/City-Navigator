# Python base image 
FROM python:3.10

# Install necessary packages, including PostgreSQL client
# RUN apt-get update && apt-get install -y \
#     postgresql-client \
#     && rm -rf /var/lib/apt/lists/*


#Set the working directory
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["fastapi", "run", "app/main.py", "--port", "80"]


# docker build -t flask-app:v1.0 .
# docker run -d -p 80:80 flask-app:v1.0 
# docker container ls
# docker conatiner stop __