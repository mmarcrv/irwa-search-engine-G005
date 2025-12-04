import json
import os
from json import JSONEncoder
from datetime import datetime

import httpagentparser  # for getting the user agent as json
from flask import Flask, redirect, render_template, session
from flask import request
import pandas as pd
from rank_bm25 import BM25Okapi

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.search.algorithms import create_index_tfidf
from myapp.analytics.database import insert_user, insert_session, insert_doc_click, update_doc_click_dwell_time, insert_log_click, insert_log_request

from myapp.search.algorithms import token_cleaning_text

from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

import pymysql

def get_db():
    return pymysql.connect(
        host=os.getenv("MYSQLHOST"),
        user=os.getenv("MYSQLUSER"),
        password=os.getenv("MYSQLPASSWORD"),
        database=os.getenv("MYSQLDATABASE"),
        port=int(os.getenv("MYSQLPORT")),
        cursorclass=pymysql.cursors.DictCursor
    )


# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

#accept the environment variable only if it points to a .zip, otherwise use the default zip. 
env = os.getenv("DATA_FILE_PATH")
if env and env.lower().endswith(".zip") and os.path.exists(os.path.join(path, env)):
    data_rel = env
else:
    data_rel = "data/cleaned_fashion_products.zip"
file_path = os.path.join(path, data_rel)
print("Using dataset file:", file_path) #to check which dataset is being used

corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
#print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])

# Convert corpus to a dataframe structure
corpus_dataframe = pd.DataFrame(
        [doc.model_dump() for doc in corpus.values()]
    )

# Create the index, tf, df and idf to avoid repeating this for every search
index_tf, tf, df, idf = create_index_tfidf(corpus_dataframe)
print("\nCreated index, tf, df and idf!")

# Create BM25 model to avoid repeating this for every search
paragraph_tokens = corpus_dataframe["cleaned_title_description_extra_fields"].tolist()
bm25 = BM25Okapi(paragraph_tokens)
print("\nBM25 search engine ready!")

# Variable to control the session creation
flag_session = False
flag_user = False

# Home URL "/"
@app.route('/')
def index():
    global flag_user, flag_session
    print("starting home url /...")
    
    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    
    user_agent = request.headers.get('User-Agent')
    print("\nRaw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    #print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if "user_id" not in session:
        conn = get_db()
        user_id = insert_user(conn, agent=agent, ip_address=user_ip, first_visit=start_time)
        conn.close()
        analytics_data.save_user_context(user_id=user_id, user_ip=user_ip, agent=agent, start_time=start_time)
        session['user_id'] = user_id
        print("New user created:", user_id)
        flag_user = True
    
    if flag_user == False:
        print("User already exists:", session['user_id'])
        conn = get_db()
        sessions = analytics_data.load_sessions(conn, session['user_id'])
        conn.close()
        print(f"Loaded {len(sessions)} previous sessions for user {session['user_id']}")
        flag_user = True
  
    if flag_session == False:
        conn = get_db()
        session_id = insert_session(conn, start_time=start_time, user_id=session['user_id'])
        conn.close()
        analytics_data.new_session(session_id=session_id, user_id=session['user_id'], start_time=start_time)
        session['session_id'] = session_id
        print("New session created:", session_id)
        flag_session = True
    
    print(session)
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    analytics_data.log_click(
        user_id=session["user_id"],
        session_id=session["session_id"],
        element="search_button",
        timestamp=timestamp
    )
    insert_log_click(conn, session_id=session["session_id"], user_id=session["user_id"], element="search_button", timestamp=timestamp)
    conn.close()

    search_query = request.form['search-query']
    selected_engine = request.form.get('engine', 'tfidf') 

    session['last_search_query'] = search_query

    query_terms = token_cleaning_text(search_query)

    results = search_engine.search(search_query, query_terms, corpus, corpus_dataframe, index_tf, tf, idf, bm25, selected_engine)
    session['last_ranked_docs'] = [doc.pid for doc in results[:20]]

    found_count = len(results)
    session['last_found_count'] = found_count

    # funció per fer el save de la query aquí en el sql!!
    conn = get_db()
    query_id, mission_id, research_mission_id = analytics_data.save_query(conn, search_query, query_terms, found_count, session['session_id'])
    conn.close()
    session['search_id'] = query_id
    
    analytics_data.save_query_terms(query_id, search_query, query_terms, found_count, session['session_id'], mission_id, research_mission_id)
    ### unir-ho amb això potser??
    #analytics_data.add_query(session['session_id'], search_id, query_terms)
    
    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)
    session['last_rag_response'] = rag_response

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count, rag_response=rag_response)

@app.route('/results', methods=['GET'])
def show_previous_results():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    analytics_data.log_click(
        user_id=session["user_id"],
        session_id=session["session_id"],
        element="return_to_results",
        timestamp=timestamp
    )
    insert_log_click(conn, session_id=session["session_id"], user_id=session["user_id"], element="return_to_results", timestamp=timestamp)
    conn.close()
    # Si no hi ha dades, torna a l’index
    if 'last_ranked_docs' not in session:
        return redirect('/')

    # Reconstruir els docs ja ordenats
    ranked_pids = session['last_ranked_docs']
    results = [corpus[pid] for pid in ranked_pids]

    found_count = session.get('last_found_count', len(results))

    return render_template(
        'results.html',
        results_list=results,
        page_title="Results",
        found_counter=found_count,
        rag_response=session.get('last_rag_response', "")
    )


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show document details page
    ### Replace with your custom logic ###
    """

    # getting request parameters:
    # user = request.args.get('user')
    print("doc details session: ")
    print(session)

    query_id = session["search_id"]
    print("Query_id:", query_id)

    # get the query string parameters from request
    clicked_doc_id = request.args["pid"]
    print("click in id={}".format(clicked_doc_id))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    analytics_data.log_click(
        user_id=session["user_id"],
        session_id=session["session_id"],
        element=f"doc_{clicked_doc_id}",
        timestamp=timestamp
    )
    insert_log_click(conn, session_id=session["session_id"], user_id=session["user_id"], element=f"doc_{clicked_doc_id}", timestamp=timestamp)
    conn.close()

    doc = corpus[clicked_doc_id]

    ranked_docs = session.get('last_ranked_docs', [])

    try:
        ranking = ranked_docs.index(clicked_doc_id) + 1
    except ValueError:
        ranking = -1

    analytics_data.save_document_click(doc_id=clicked_doc_id, query_id=query_id, ranking=ranking)
    conn = get_db()
    insert_doc_click(conn, doc_pid=clicked_doc_id, query_id=query_id, ranking=ranking)
    conn.close()

    print("Current document clicks table:")
    print(analytics_data.document_clicks_table)

    return render_template('doc_details.html', doc=doc)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    analytics_data.log_click(
        user_id=session["user_id"],
        session_id=session["session_id"],
        element=f"stats",
        timestamp=timestamp
    )
    insert_log_click(conn, session_id=session["session_id"], user_id=session["user_id"], element=f"stats", timestamp=timestamp)
    conn.close()
    """
    docs = []
    for doc_id, clicks_list in analytics_data.document_clicks_table.items():
        row: Document = corpus[doc_id]
        count = len(clicks_list)
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)
    """

    conn = get_db()
    try:
        with conn.cursor() as cur:
            # Indicadors clau
            cur.execute("SELECT COUNT(*) AS total FROM users")
            num_users = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(*) AS total FROM sessions")
            num_sessions = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(*) AS total FROM queries")
            num_queries = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(*) AS total FROM doc_click")
            num_doc_clicks = cur.fetchone()["total"]

            cur.execute("SELECT ROUND(AVG(dwell_time_minutes),2) AS avg_dwell FROM doc_click")
            avg_dwell_time = cur.fetchone()["avg_dwell"]

            # Comptar clics per document
            cur.execute("""
                SELECT doc_pid, COUNT(*) AS num_clicks
                FROM doc_click
                GROUP BY doc_pid
                ORDER BY num_clicks DESC
            """)
            rows = cur.fetchall()

            docs = []
            for r in rows:
                doc_id = r["doc_pid"]
                count = r["num_clicks"]
                # Suposant que tens corpus[doc_id] amb metadades
                row = corpus[doc_id]
                doc = StatsDocument(
                    pid=row.pid,
                    title=row.title,
                    description=row.description,
                    url=row.url,
                    count=count
                )
                docs.append(doc)

    finally:
        conn.close()

    return render_template(
        'stats.html',
        clicks_data=docs,
        num_users=num_users,
        num_sessions=num_sessions,
        num_queries=num_queries,
        num_doc_clicks=num_doc_clicks,
        avg_dwell_time=avg_dwell_time
    )


@app.route('/dashboard', methods=['GET'])
def dashboard():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    analytics_data.log_click(
        user_id=session["user_id"],
        session_id=session["session_id"],
        element=f"dashboard",
        timestamp=timestamp
    )
    insert_log_click(conn, session_id=session["session_id"], user_id=session["user_id"], element=f"dashboard", timestamp=timestamp)
    conn.close()

    """
    visited_docs = []
    for doc_id, clicks_list in analytics_data.document_clicks_table.items():
        d: Document = corpus[doc_id]
        count = len(clicks_list)
        doc = ClickedDoc(doc_id, d.description, count)
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)
    """
    conn = get_db()
    try:
        with conn.cursor() as cur:
            # Consultes per dia
            cur.execute("""
                SELECT DATE(timestamp) AS day, COUNT(*) AS num_queries
                FROM queries
                GROUP BY day
                ORDER BY day
            """)
            rows = cur.fetchall()
            queries_per_day_labels = [r["day"].strftime("%Y-%m-%d") for r in rows]
            queries_per_day_counts = [r["num_queries"] for r in rows]

            # Navegadors
            cur.execute("SELECT browser, COUNT(*) AS total FROM users GROUP BY browser")
            browsers = cur.fetchall()

            # Sistemes operatius
            cur.execute("SELECT os, COUNT(*) AS total FROM users GROUP BY os")
            os_data = cur.fetchall()

            # Clics retornats
            cur.execute("""
                SELECT returned_to_results, COUNT(*) AS total
                FROM doc_click
                GROUP BY returned_to_results
            """)
            clicks_returned = cur.fetchall()

    finally:
        conn.close()

    return render_template(
        'dashboard.html',
        queries_per_day_labels=queries_per_day_labels,
        queries_per_day_counts=queries_per_day_counts,
        browsers=browsers,
        os_data=os_data,
        clicks_returned=clicks_returned
    )


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    analytics_data.log_click(
        user_id=session["user_id"],
        session_id=session["session_id"],
        element=f"plot_number_of_views",
        timestamp=timestamp
    )
    insert_log_click(conn, session_id=session["session_id"], user_id=session["user_id"], element=f"plot_number_of_views", timestamp=timestamp)
    conn.close()
    return analytics_data.plot_number_of_views()

@app.route('/log_dwell_time', methods=['POST'])
def log_dwell_time():
    data = json.loads(request.data.decode("utf-8"))

    doc_id = data["doc_id"]
    query_id = data["query_id"]
    dwell_time_ms = data["dwell_time_ms"]
    returned_via_results = data["returned_via_results"]

    analytics_data.update_dwell_time(doc_id, query_id, dwell_time_ms)
    conn = get_db()
    update_doc_click_dwell_time(conn, query_id, doc_id, (dwell_time_ms / 60000), returned_via_results)
    conn.close()

    return "", 204   # resposta mínima per sendBeacon

@app.before_request
def track_request():
    if "user_id" in session and "session_id" in session:
        analytics_data.log_request(
            user_id=session["user_id"],
            session_id=session["session_id"],
            method=request.method,
            url=request.path,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        conn = get_db()
        insert_log_request(
            conn,
            user_id=session["user_id"],
            session_id=session["session_id"],
            method=request.method,
            url=request.path,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        conn.close()

@app.route('/reset_session')
def reset_session():
    session.clear()
    print("Session cleared. Next visit will create a new session.")
    return "Session cleared. Next visit will create a new session."

if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
