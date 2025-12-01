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
    Analytics data manager that works with external MySQL database
    and maintains some in-memory data for the current session
    """
    
    # In-memory storage for current session
    sessions = []
    query_scores = {}  # Store scores for queries: {query_id: {doc_pid: score}}
    
    def load_sessions(self, conn, user_id):
        """Load all sessions for a user from the database"""
        raw_sessions = get_sessions(conn, user_id)
        
        sessions = []
        for session in raw_sessions:
            session_id = session["session_id"]
            
            # Load queries for this session
            queries = get_queries_by_session(conn, session_id)
            
            # Group queries into missions based on similarity
            missions = self._group_queries_into_missions(queries)
            
            sessions.append({
                "session_id": session_id,
                "start_time": session["start_time"],
                "end_time": session.get("end_time"),
                "user_id": session["user_id"],
                "queries": queries,
                "missions": missions
            })
        
        return sessions

    def _group_queries_into_missions(self, queries):
        """Group queries into missions based on similarity"""
        if not queries:
            return []
        
        missions = []
        THRESHOLD = 0.3
        
        for query in queries:
            query_id = query["query_id"]
            query_text = query["query"]
            query_terms = set(query_text.lower().split())
            
            # Find best matching mission
            best_sim = -1
            best_mission = None
            
            for mission in missions:
                # Get last query in mission
                last_query_id = mission["query_ids"][-1]
                last_query = next((q for q in queries if q["query_id"] == last_query_id), None)
                
                if last_query:
                    last_terms = set(last_query["query"].lower().split())
                    
                    # Calculate Jaccard similarity
                    intersection = query_terms.intersection(last_terms)
                    union = query_terms.union(last_terms)
                    sim = len(intersection) / len(union) if union else 0.0
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_mission = mission
            
            # Add to existing mission or create new one
            if best_sim >= THRESHOLD and best_mission:
                best_mission["query_ids"].append(query_id)
            else:
                mission_id = len(missions) + 1
                missions.append({
                    "mission_id": mission_id,
                    "query_ids": [query_id]
                })
        
        return missions

    def new_session(self, session_id, user_id, start_time):
        """Create a new session in memory"""
        new_session = {
            "session_id": session_id,
            "start_time": start_time,
            "user_id": user_id,
            "queries": [],
            "missions": []
        }
        
        self.sessions.append(new_session)
        print("[DEBUG] Created new session:", new_session)
        
        return session_id

    def save_query(self, conn, search_query, query_terms, num_results, session_id):
        """Save query to database and return query_id"""
        num_terms = len(query_terms)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine mission_id based on similarity with previous queries
        mission_id = self._determine_mission_id(conn, session_id, query_terms)
        
        # Insert into database
        query_id = insert_query(
            conn, 
            query=search_query,
            num_terms=num_terms,
            num_results=num_results,
            timestamp=timestamp,
            session_id=session_id,
            mission_id=mission_id
        )
        
        print(f"Saved query {query_id} to database with mission_id={mission_id}")
        
        return query_id

    def _determine_mission_id(self, conn, session_id, query_terms):
        """Determine which mission this query belongs to"""
        # Get recent queries from this session
        queries = get_queries_by_session(conn, session_id)
        
        if not queries:
            return 1  # First mission
        
        # Get the last query's mission_id
        last_query = queries[-1]
        last_mission_id = last_query.get("mission_id", 1)
        
        # Calculate similarity with last query
        last_query_terms = set(last_query["query"].lower().split())
        current_terms = set(query_terms)
        
        intersection = last_query_terms.intersection(current_terms)
        union = last_query_terms.union(current_terms)
        similarity = len(intersection) / len(union) if union else 0.0
        
        THRESHOLD = 0.3
        
        if similarity >= THRESHOLD:
            return last_mission_id
        else:
            return last_mission_id + 1

    def save_document_click(self, conn, doc_id, query_id, ranking):
        """Save document click to database"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        insert_doc_click(
            conn,
            query_id=query_id,
            doc_pid=doc_id,
            ranking=ranking,
            dwell_time_minutes=None,
            timestamp=timestamp,
            returned_to_results=False
        )
        
        print(f"Saved click event for doc {doc_id}, query {query_id}")

    def update_dwell_time(self, conn, doc_id, query_id, dwell_time_ms):
        """Update dwell time for a document click"""
        # Convert milliseconds to minutes
        dwell_time_minutes = round(dwell_time_ms / 60000, 2)
        
        update_doc_click_dwell_time(conn, query_id, doc_id, dwell_time_minutes)
        
        print(f"Updated dwell time for doc {doc_id}, query {query_id}: {dwell_time_minutes} min")
        return True

    def get_document_clicks(self, conn, doc_pid):
        """Get all clicks for a specific document from database"""
        return get_doc_clicks(conn, doc_pid)

    def save_query_scores(self, query_id, doc_scores):
        """Save the scores for documents in a query result"""
        self.query_scores[query_id] = doc_scores
        print(f"Saved scores for query {query_id}: {len(doc_scores)} documents")

    def get_query_score(self, query_id, doc_pid):
        """Get the score for a specific document in a query"""
        if query_id in self.query_scores:
            return self.query_scores[query_id].get(doc_pid)
        return None

    def plot_number_of_views(self):
        # Prepare data
        data = [{'Document ID': doc_id, 'Number of Views': len(click_list)} 
                for doc_id, click_list in self.document_clicks_table.items()]
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

