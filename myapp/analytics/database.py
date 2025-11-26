def insert_user(conn, browser, os, ip_address, first_visit, last_visit=None):
    cursor = conn.cursor()

    sql = """
        INSERT INTO users (browser, os, ip_address, first_visit, last_visit)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING user_id;
    """

    cursor.execute(sql, (browser, os, ip_address, first_visit, last_visit))
    
    # Get the generated user_id from the RETURNING clause
    user_id = cursor.fetchone()[0]
    
    conn.commit()
    cursor.close()
    
    return user_id
    
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