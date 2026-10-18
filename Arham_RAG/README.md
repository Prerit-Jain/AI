Installation of Postgresql database is done by creating docker container. 
Installation Steps are mentioned in https://github.com/pgvector/pgvector#installation-notes---linux-and-mac

F:\pgvector>docker run --name pgvector-container1 -e POSTGRES_PASSWORD=root -e POSTGRES_DB=ragdb  -p 5432:5432 -d pgvector/pgvector:pg16

F:\pgvector>docker ps
CONTAINER ID   IMAGE                    COMMAND                  CREATED          STATUS          PORTS                                         NAMES
63eaa665cafd   pgvector/pgvector:pg16   "docker-entrypoint.s…"   11 seconds ago   Up 10 seconds   0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp   pgvector-container1

docker start pgvector-container1
docker exec -it pgvector-container1 psql -U postgres -d ragdb

CREATE EXTENSION IF NOT EXISTS vector;

ragdb=# \dt
           List of relations
 Schema |   Name    | Type  |  Owner
--------+-----------+-------+----------
 public | documents | table | postgres
(1 row)

The RAG application will run on Vscode in windows only.
