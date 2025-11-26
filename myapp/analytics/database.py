from datetime import datetime

def insert_user(conn, agent, ip_address, first_visit, last_visit=None):
    cursor = conn.cursor()

    browser = agent.get("browser", {}).get("name", "Unknown")
    os = agent.get("platform", {}).get("name", "Unknown")
    ip_address = ip_address
    
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


def insert_session(conn, start_time, user_id):
    cursor = conn.cursor()

    sql = """
        INSERT INTO session (start_time, user_id)
        VALUES (%s, %s)
    """

    cursor.execute(sql, (start_time, user_id))
    
    # Get the generated user_id from the RETURNING clause
    session_id = cursor.lastrowid

    
    conn.commit()
    cursor.close()
    
    return session_id



""" 
def fetch_queries(conn, limit=5):
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
"""  