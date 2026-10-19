import psycopg2 as pg
import datetime

# https://www.psycopg.org/docs/usage.html

conn = pg.connect(
        user = "vit",
        password = None,
        host = "/var/run/postgresql",
        port = "5432")

c = conn.cursor()

# Создание таблицы (DDL)
c.execute("CREATE TABLE IF NOT EXISTS obs(dtm TIMESTAMP, oname TEXT);")
conn.commit()

# Заполнение таблицы (DML)
c.execute("INSERT INTO obs(dtm, oname) VALUES (%s, %s)", (
    datetime.datetime.now(),
    "SS433"
    ))
conn.commit()

c.execute("SELECT dtm, oname FROM obs;")
for tup in c:
    print(tup[0], type(tup[0]), tup[1], type(tup[1]))

c.close()
conn.close()

