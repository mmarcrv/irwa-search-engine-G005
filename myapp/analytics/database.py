from datetime import datetime


def insert_log_request(conn, session_id, user_id, method, url, timestamp):
    cursor = conn.cursor()

    # Obtenir el següent request_id per la sessió
    cursor.execute("SELECT MAX(request_id) FROM log_request WHERE session_id = %s", (session_id,))
    result = cursor.fetchone()
    next_request_id = (result[0] or 0) + 1

    sql = """
        INSERT INTO log_request (request_id, session_id, user_id, method, url, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (next_request_id, session_id, user_id, method, url, timestamp))
    conn.commit()
    cursor.close()

    return next_request_id


def insert_log_click(conn, session_id, user_id, element, timestamp):
    cursor = conn.cursor()

    # Obtenir el següent click_id per la sessió
    cursor.execute("SELECT MAX(click_id) FROM log_click WHERE session_id = %s", (session_id,))
    result = cursor.fetchone()
    next_click_id = (result[0] or 0) + 1

    sql = """
        INSERT INTO log_click (click_id, session_id, user_id, element, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (next_click_id, session_id, user_id, element, timestamp))
    conn.commit()
    cursor.close()

    return next_click_id

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
        INSERT INTO sessions (start_time, user_id, missions)
        VALUES (%s, %s, %s)
    """

    cursor.execute(sql, (start_time, user_id, 0))
    
    # Get the generated user_id from the RETURNING clause
    session_id = cursor.lastrowid

    
    conn.commit()
    cursor.close()
    
    return session_id



def update_session_end_time(conn, session_id, end_time):
    """Update the end_time for a session"""
    cursor = conn.cursor()
    
    sql = """
        UPDATE sessions 
        SET end_time = %s 
        WHERE session_id = %s
    """
    
    cursor.execute(sql, (end_time, session_id))
    conn.commit()
    cursor.close()


def get_sessions(conn, user_id):
    cursor = conn.cursor()

    sql = """
        SELECT * FROM sessions WHERE user_id = %s ORDER BY start_time DESC
    """

    cursor.execute(sql, (user_id,))
    
    sessions = cursor.fetchall()
    cursor.close()
    
    return sessions


def insert_query(conn, query, query_terms, num_terms, num_results, timestamp, session_id, mission_id, research_mission_id):
    """Insert a new query into the database"""
    cursor = conn.cursor()
    
    sql = """
        INSERT INTO queries (query, query_terms, num_terms, num_results, timestamp, session_id, mission_id, research_mission_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    query_terms_str = " ".join(query_terms) if isinstance(query_terms, (list, set)) else str(query_terms)

    cursor.execute(sql, (query, query_terms_str, num_terms, num_results, timestamp, session_id, mission_id, research_mission_id))
    query_id = cursor.lastrowid
    
    conn.commit()
    cursor.close()
    
    return query_id


def get_queries_by_session(conn, session_id):
    """Get all queries for a specific session"""
    cursor = conn.cursor()
    
    sql = """
        SELECT * FROM queries WHERE session_id = %s ORDER BY timestamp ASC
    """
    
    cursor.execute(sql, (session_id,))
    queries = cursor.fetchall()
    cursor.close()
    
    return queries


def insert_doc_click(conn, query_id, doc_pid, ranking, dwell_time_minutes=None, timestamp=None, returned_to_results=False):
    """Insert a document click event"""
    cursor = conn.cursor()
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    sql = """
        INSERT INTO doc_click (query_id, doc_pid, ranking, dwell_time_minutes, timestamp, returned_to_results)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    
    cursor.execute(sql, (query_id, doc_pid, ranking, dwell_time_minutes, timestamp, returned_to_results))
    
    conn.commit()
    cursor.close()


def update_doc_click_dwell_time(conn, query_id, doc_pid, dwell_time_minutes, returned_to_results):
    """Update the dwell time for a document click"""
    cursor = conn.cursor()
    
    sql = """
        UPDATE doc_click 
        SET dwell_time_minutes = %s, returned_to_results = %s 
        WHERE query_id = %s AND doc_pid = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """
    
    cursor.execute(sql, (dwell_time_minutes, returned_to_results, query_id, doc_pid))
    conn.commit()
    cursor.close()


def get_doc_clicks(conn, doc_pid):
    """Get all clicks for a specific document"""
    cursor = conn.cursor()
    
    sql = """
        SELECT * FROM doc_click WHERE doc_pid = %s ORDER BY timestamp DESC
    """
    
    cursor.execute(sql, (doc_pid,))
    clicks = cursor.fetchall()
    cursor.close()
    
    return clicks


def get_all_doc_clicks(conn):
    """Get all document clicks for analytics"""
    cursor = conn.cursor()
    
    sql = """
        SELECT doc_pid, COUNT(*) as click_count
        FROM doc_click
        GROUP BY doc_pid
        ORDER BY click_count DESC
    """
    
    cursor.execute(sql)
    clicks = cursor.fetchall()
    cursor.close()
    
    return clicks