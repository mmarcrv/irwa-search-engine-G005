from datetime import datetime
import json
import altair as alt
import pandas as pd
from myapp.analytics.database import (
    get_sessions, get_queries_by_session, insert_query, 
    insert_doc_click, update_doc_click_dwell_time, get_doc_clicks
)

class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # Example of statistics table
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    #fact_clicks = dict([])
    document_clicks_table = {}

    ### Please add your custom tables here:
    sessions = []
    user_queries = []
    user_table = {}
    requests_table = []
    clicks_table = []

    def log_request(self, user_id, session_id, method, url, timestamp):
        entry = {
            "user_id": user_id,
            "session_id": session_id,
            "method": method,
            "url": url,
            "timestamp": timestamp
        }
        self.requests_table.append(entry)
        print("[DEBUG] Logged request:", entry)
    
    def log_click(self, user_id, session_id, element, timestamp):
        entry = {
            "user_id": user_id,
            "session_id": session_id,
            "element": element,
            "timestamp": timestamp
        }
        self.clicks_table.append(entry)
        print("[DEBUG] Logged click:", entry)

    def new_session(self, session_id, user_id, start_time):

        new_session = {
            "session_id": session_id,
            "start_time": start_time,
            "user_id": user_id,
            "queries": [],
            "missions": 0
        }

        self.sessions.append(new_session)
        print("[DEBUG] Created new session:", new_session)
        print("[DEBUG] Current sessions listed:", len(self.sessions))

        return session_id

    def load_sessions(self, conn, user_id):
        """Load all sessions for a user from the database"""
        raw_sessions = get_sessions(conn, user_id)
        for s in raw_sessions:
            session_id = s["session_id"]
            
            # Load queries for this session
            queries = get_queries_by_session(conn, session_id)
            for q in queries:
                query = {
                    "query_id": q["id"],
                    "query": q["query"],
                    "query_terms": q["query_terms"],
                    "num_terms": q["num_terms"],
                    "num_results": q["num_results"],
                    "timestamp": q["timestamp"],
                    "session_id": q["session_id"],
                    "mission_id": q["mission_id"],
                    "research_mission_id": q["research_mission_id"]
                }
                self.user_queries.append(query)
            
            # Group queries into missions based on similarity
            self.sessions.append({
                "session_id": session_id,
                "start_time": s["start_time"],
                "end_time": s.get("end_time"),
                "user_id": s["user_id"],
                "queries": queries,
                "missions": s['missions']
            })
        
        return self.sessions

    # def _group_queries_into_missions(self, queries):
    #     """Group queries into missions based on similarity"""
    #     if not queries:
    #         return []
        
    #     missions = []
    #     THRESHOLD = 0.3
        
    #     for query in queries:
    #         query_id = query["id"]
    #         for q in queries:
    #             print("Query:", q["query"])
    #         query_terms = set(query["query_terms"])
    #         for q in queries:
    #             print("Query:", q["query_terms"])
            
    #         # Find best matching mission
    #         best_sim = -1
    #         best_mission = None
            
    #         for mission in missions:
    #             # Get last query in mission
    #             last_query_id = mission["query_ids"][-1]
    #             last_query = next((q for q in queries if q["id"] == last_query_id), None)
                
    #             if last_query:
    #                 last_terms = set(last_query["query_terms"])
                    
    #                 # Calculate Jaccard similarity
    #                 intersection = query_terms.intersection(last_terms)
    #                 union = query_terms.union(last_terms)
    #                 sim = len(intersection) / len(union) if union else 0.0
                    
    #                 if sim > best_sim:
    #                     best_sim = sim
    #                     best_mission = mission
            
    #         # Add to existing mission or create new one
    #         if best_sim >= THRESHOLD and best_mission:
    #             best_mission["query_ids"].append(query_id)
    #         else:
    #             mission_id = len(missions) + 1
    #             missions.append({
    #                 "mission_id": mission_id,
    #                 "query_ids": [query_id]
    #             })
        
    #     return missions
    
    def add_query(self, session_id, query_id, query_terms):
        print("[DEBUG] add_query called with session_id:", session_id)
        print("[DEBUG] Current sessions:", self.sessions)
        
        for s in self.sessions:
            print("[DEBUG] checking session:", s["session_id"])
            if int(s["session_id"]) == int(session_id):
                print("[DEBUG] MATCH FOUND for session_id:", session_id)
                # --- 1) Afegir query ---
                s["queries"].append({
                    "query_id": query_id,
                    "query_terms": query_terms
                })
                print(f"Added query {query_id} to session {session_id}")


                # --- 2) Assegurar que existeix missions ---
                if "missions" not in s:
                    s["missions"] = []

                # --- 3) Si no hi ha cap missió → crear la primera ---
                if len(s["missions"]) == 0:
                    s["missions"].append({
                        "mission_id": 1,
                        "query_ids": [query_id]
                    })
                    print(f"Added query {query_id} to NEW mission 1 in session {session_id}")
                    return True
                
                # --- 4) Comparar amb totes les missions existents ---
                best_sim = -1
                best_mission = None
                for mission in s["missions"]:
                    last_query_id = mission["query_ids"][-1]
                    last_query_terms = None
                    for q in s["queries"]:
                        if q["query_id"] == last_query_id:
                            last_query_terms = q["query_terms"]
                            break
                    
                    s1 = set(last_query_terms)
                    s2 = set(query_terms)
                    intersection = s1.intersection(s2)
                    union = s1.union(s2)
                    if not union:
                        sim = 0.0
                    else:
                        sim = len(intersection) / len(union)

                    if sim > best_sim:
                        best_sim = sim
                        best_mission = mission
            
                # --- 5) Decisió: afegir a missió existent o crear-ne una nova ---
                THRESHOLD = 0.3

                if best_sim >= THRESHOLD:
                    best_mission["query_ids"].append(query_id)
                    print(f"Added query {query_id} to mission {best_mission['mission_id']} (sim={best_sim:.2f})")
                else:
                    new_id = s["missions"][-1]["mission_id"] + 1
                    s["missions"].append({
                        "mission_id": new_id,
                        "query_ids": [query_id]
                    })
                    print(f"Created NEW mission {new_id} for query {query_id} (best_sim={best_sim:.2f})")

                return True

        print("Session not found:", session_id)
        return False

    

    def save_query_terms(self, query_id, search_query, query_terms, num_results, session_id, mission_id, research_mission_id):
        num_terms = len(query_terms)

        query = {
            "query_id": query_id,
            "query": search_query,
            "query_terms": query_terms,
            "num_terms": num_terms,
            "num_results": num_results,
            "timestamp": pd.Timestamp.now(),
            "session_id": session_id,
            "mission_id": mission_id,
            "research_mission_id": research_mission_id  
        }
        self.user_queries.append(query)

        print("Saved query:", query["query_id"])

        return query_id
    
    def save_query(self, conn, search_query, query_terms, num_results, session_id):
        """Save query to database and return query_id"""
        num_terms = len(query_terms)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine mission_id based on similarity with previous queries
        mission_id = self.determine_mission_id(session_id, query_terms)
        research_mission_id = self.determine_research_mission_id(session_id, query_terms)
        
        # Insert into database
        query_id = insert_query(
            conn, 
            query=search_query,
            query_terms=query_terms,
            num_terms=num_terms,
            num_results=num_results,
            timestamp=timestamp,
            session_id=session_id,
            mission_id=mission_id,
            research_mission_id=research_mission_id
        )
        
        print(f"Saved query {query_id} to database with mission_id={mission_id}")
        
        return query_id, mission_id, research_mission_id

    def determine_mission_id(self, session_id, query_terms):
        """Determine which mission this query belongs to"""
        # Get recent queries from this session
        missions = None
        for s in self.sessions:
            if s["session_id"] == session_id:
                missions = s["missions"]
                break
        
        if missions == None:
            for s in self.sessions:
                if s["session_id"] == session_id:
                    s["missions"] = 1
                    break
            print(f"Added query to NEW mission 1 in session {session_id}")
            return 1
        
        # --- 4) Comparar amb totes les missions existents ---
        best_sim = -1
        best_mission = None
        for mission_id in range(missions):
            sim = 0
            queries_mission = [q for q in self.user_queries if q["session_id"] == session_id and q["mission_id"] == mission_id+1]
            query_terms_mission = None
            for q in queries_mission:
                query_terms_mission = q["query_terms"]                
                s1 = set(query_terms_mission)
                s2 = set(query_terms)
                intersection = s1.intersection(s2)
                union = s1.union(s2)
                if not union:
                    sim = 0.0
                else:
                    sim += len(intersection) / len(union)

            sim /= len(queries_mission)
            if sim > best_sim:
                best_sim = sim
                best_mission = mission_id
    
        # --- 5) Decisió: afegir a missió existent o crear-ne una nova ---
        THRESHOLD = 0.3

        if best_sim >= THRESHOLD:
            print(f"Added query to mission {best_mission+1} (sim={best_sim:.2f})")
            return best_mission+1
        else:
            new_id = missions+1
            for s in self.sessions:
                if s["session_id"] == session_id:
                    s["missions"] = new_id
                    break
            print(f"Created NEW mission {new_id} for query (best_sim={best_sim:.2f})")
            return new_id

    
    def determine_research_mission_id(self, session_id, query_terms):

        if len(self.user_queries) == 0:
            return 1

        num_research_missions = max(q["research_mission_id"] for q in self.user_queries)

        print(f"\nCurrent number of research missions: {num_research_missions}\n")
        
        # --- 4) Comparar amb totes les missions existents ---
        best_sim = -1
        best_mission = None
        for mission_id in range(num_research_missions):
            sim = 0
            queries_mission = [q for q in self.user_queries if q["research_mission_id"] == mission_id+1]
            query_terms_mission = None
            for q in queries_mission:
                query_terms_mission = q["query_terms"]                
                s1 = set(query_terms_mission)
                s2 = set(query_terms)
                intersection = s1.intersection(s2)
                union = s1.union(s2)
                if not union:
                    sim = 0.0
                else:
                    sim += len(intersection) / len(union)

            sim /= len(queries_mission)
            if sim > best_sim:
                best_sim = sim
                best_mission = mission_id
    
        # --- 5) Decisió: afegir a missió existent o crear-ne una nova ---
        THRESHOLD = 0.3

        if best_sim >= THRESHOLD:
            print(f"Added query to mission {best_mission+1} (sim={best_sim:.2f})")
            return best_mission+1
        else:
            new_id = num_research_missions+1
            print(f"Created NEW mission {new_id} for query (best_sim={best_sim:.2f})")
            return new_id
    
    def save_user_context(self, user_id, user_ip, agent, start_time):

        browser = agent.get("browser", {}).get("name", "Unknown")
        os = agent.get("platform", {}).get("name", "Unknown")
        ip = user_ip

        self.user_table[user_id] = {
            "browser": browser,
            "os": os,
            "ip": ip,
            "timestamp": start_time,
        }

        print("Saved user:", self.user_table[user_id])

        return user_id

    def save_document_click(self, doc_id, query_id, ranking):

        if doc_id not in self.document_clicks_table:
            self.document_clicks_table[doc_id] = []

        doc_click = {
            "query_id": query_id,
            "ranking": ranking,
            "dwell_time": -1,  # Placeholder
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.document_clicks_table[doc_id].append(doc_click)

        print(f"Saved click event for doc {doc_id}:", doc_click)

    def update_dwell_time(self, doc_id, query_id, dwell_time_ms):
        clicks = self.document_clicks_table.get(doc_id, [])

        for click in reversed(clicks):
            print(click["query_id"])
            print(query_id)
            print(click["dwell_time"])
            print(dwell_time_ms)

            if int(click["query_id"]) == int(query_id) and int(click["dwell_time"]) == -1:
                click["dwell_time"] = dwell_time_ms
                print("Dwell time updated:", click)
                return True

        print("No click found to update dwell time")
        return False

    
    def plot_number_of_views(self):
        # Prepare data
        data = [{'Document ID': doc_id, 'Number of Views': len(click_list)} for doc_id, click_list in self.document_clicks_table.items()]
        df = pd.DataFrame(data)
        # Create Altair chart
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID',
            y='Number of Views'
        ).properties(
            title='Number of Views per Document'
        )
        # Render the chart to HTML
        return chart.to_html()


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
