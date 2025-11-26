from datetime import datetime
import json
import altair as alt
import pandas as pd

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
    query_table = {}
    counter_query_id = 1
    sessions = []
    user_table = {}

    def new_session(self, session_id, user_id):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_session = {
            "session_id": session_id,
            "start_time": timestamp,
            "user_id": user_id,
            "queries": [],
            "missions": []
        }

        self.sessions.append(new_session)
        print("[DEBUG] Created new session:", new_session)
        print("[DEBUG] Current sessions list:", self.sessions)

        return session_id

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

    

    def save_query_terms(self, search_query, query_terms, num_results):
        num_terms = len(query_terms)

        query_id = self.counter_query_id
        self.counter_query_id += 1

        self.query_table[query_id] = {
            "search_query": search_query,
            "order": query_terms,
            "num_terms": num_terms,
            "num_results": num_results,
            "timestamp": pd.Timestamp.now()
        }

        print("Saved query:", self.query_table[query_id])

        return query_id
    
    def save_user_context(self, user_id, session_id, user_ip, agent):

        browser = agent.get("browser", {}).get("name", "Unknown")
        os = agent.get("platform", {}).get("name", "Unknown")
        ip = user_ip
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.user_table[user_id] = {
            "session_id": session_id,
            "browser": browser,
            "os": os,
            "ip": ip,
            "timestamp": timestamp,
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
