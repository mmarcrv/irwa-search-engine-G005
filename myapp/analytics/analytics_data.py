from datetime import datetime
import json
import random
import altair as alt
import pandas as pd

class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # Example of statistics table
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict([])

    ### Please add your custom tables here:
    query_table = {}
    counter_query_id = 1
    sessions = []
    counter_user_id = 1
    user_table = {}

    def new_session(self):

        existing_ids = sorted([s["session_id"] for s in self.sessions])

        new_id = 1
        for sid in existing_ids:
            if sid == new_id:
                new_id += 1
            else:
                break

        new_session = {
            "session_id": new_id,
            "start_time": datetime.now()
        }

        self.sessions.append(new_session)

        return new_id
    

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
    
    def save_user_context(self, session_id, user_ip, agent):
        
        user_id = self.counter_user_id
        self.counter_user_id += 1

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
    
    def plot_number_of_views(self):
        # Prepare data
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
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
