FROM python:3.9-slim-bullseye

RUN pip install gunicorn==20.1.0
RUN pip install typing-extensions
RUN pip install embedding_explorer==0.3.0
RUN pip install gensim==4.2.0

COPY main.py main.py

EXPOSE 8080
CMD gunicorn --timeout 0 -b 0.0.0.0:8080 --worker-tmp-dir /dev/shm --workers=2 --threads=4 --worker-class=gthread main:server
