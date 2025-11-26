import mysql.connector
import os
from dotenv import load_dotenv
from datetime import datetime

def insert_user(conn, browser, os, ip_address, first_visit, last_visit=None):
    cursor = conn.cursor()

    sql = """
        INSERT INTO users (browser, os, ip, first_seen, last_seen)
        VALUES (%s, %s, %s, %s, %s)
    """

    cursor.execute(sql, (browser, os, ip_address, first_visit, last_visit))
    
    # Get the last inserted ID - MySQL way
    user_id = cursor.lastrowid
    
    conn.commit()
    cursor.close()
    
    return user_id

# DATABASE SETUP
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

conn = get_db()
first_visit = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
user_id = insert_user(conn, "Chrome", "IOS", "170.0.0.0", first_visit=first_visit)
print("Inserted user with ID:", user_id)
conn.close()