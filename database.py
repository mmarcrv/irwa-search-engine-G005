import os
import mysql.connector
from dotenv import load_dotenv

def get_db():
    load_dotenv() 
    conn = mysql.connector.connect(
        host=os.getenv("MYSQLHOST"),
        user=os.getenv("MYSQLUSER"),
        password=os.getenv("MYSQLPASSWORD"),
        database=os.getenv("MYSQLDATABASE"),
        port=int(os.getenv("MYSQLPORT"))
    )
    return conn


def insert_query(number_of_terms, user):
    conn = get_db()
    cursor = conn.cursor()

    sql = """
        INSERT INTO search_queries (query_id, number_of_terms, user)
        VALUES (%s, %s, %s)
    """

    cursor.execute(sql, (2, 24, "Pau"))
    conn.commit()

    print("Inserted! ID =", cursor.lastrowid)

    cursor.close()
    conn.close()

    
def fetch_queries(limit=5):
    conn = get_db()
    cursor = conn.cursor()

    # Fetch some rows from search_queries table
    cursor.execute(f"SELECT query_id, number_of_terms, user FROM search_queries LIMIT {limit}")
    rows = cursor.fetchall()

    print(f"Fetched {len(rows)} rows:")
    for row in rows:
        print(row)

    cursor.close()
    conn.close()

# Run the test
fetch_queries()