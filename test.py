import mysql.connector
import os
from dotenv import load_dotenv
from datetime import datetime

def insert_user():
    cursor = conn.cursor()

    sql = """
        CREATE TABLE log_request (
            request_id INT AUTO_INCREMENT PRIMARY KEY,
            session_id INT,
            user_id INT,
            method VARCHAR(10),
            url TEXT,
            timestamp DATETIME
        );
    """

    cursor.execute(sql)
    
    # Get the last inserted ID - MySQL way
    
    conn.commit()
    cursor.close()
    
    return 1

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
user_id = insert_user()
print("Inserted user with ID:", user_id)
conn.close()