import streamlit as st
import time
import json
import logging
import os
from typing import Dict, List, Any, Optional

from firebase_admin import credentials, initialize_app, firestore
from firebase_admin import auth as firebase_auth
from agent_core import AgentCore
from curriculum_manager import CurriculumManager

# --- CONFIGURATION & LOGGING ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_MODEL = 'gemini-2.5-flash-preview-09-2025' 
PASS_THRESHOLD = 0.70 

# --- FIREBASE INTEGRATION (P1: Persistence & LTM) ---

def init_firebase() -> tuple[Optional[firestore.client], str]:
    """Initializes Firebase Admin SDK and authenticates the user."""
    if 'firebase_initialized' in st.session_state and st.session_state.firebase_initialized:
        db = firestore.client()
        return db, st.session_state.user_id

    try:
        app_id = os.environ.get('__app_id', 'default-app-id')
        firebase_config_str = os.environ.get('__firebase_config')
        auth_token = os.environ.get('__initial_auth_token')
        
        if not firebase_config_str:
            return None, "anonymous"

        firebase_config = json.loads(firebase_config_str)
        
        if not firestore._apps:
            cred = credentials.Certificate(firebase_config)
            initialize_app(cred)

        db = firestore.client()
        
        user_id = "anonymous"
        if auth_token:
            try:
                decoded_token = firebase_auth.verify_id_token(auth_token)
                user_id = decoded_token['uid']
            except Exception as e:
                logger.error("Failed to verify auth token: %s", e)

        st.session_state.app_id = app_id
        st.session_state.user_id = user_id
        st.session_state.firebase_initialized = True
        return db, user_id
        
    except Exception as e:
        logger.error("Fatal Firebase Initialization Error: %s", e)
        return None, None

def get_session_doc_ref(db: firestore.client, user_id: str, topic: str):
    """Returns the document reference for the current curriculum session."""
    app_id = st.session_state.get('app_id', 'default-app-id')
    topic_doc_id = topic.lower().replace(' ', '_').replace('/', '_')
    collection_path = f"artifacts/{app_id}/users/{user_id}/curriculum_sessions"
    return db.collection(collection_path).document(topic_doc_id)

def load_curriculum_state(db: firestore.client, user_id: str, topic: str) -> Optional[CurriculumManager]:
    """Loads curriculum state from Firestore."""
    try:
        doc_ref = get_session_doc_ref(db, user_id, topic)
        doc = doc_ref.get()
        if doc.exists:
            logger.info("P1 Trace: Loaded state from Firestore.")
            return CurriculumManager.deserialize(doc.to_dict())
        return None
    except Exception as e:
        logger.error("Failed to load state from Firestore: %s", e)
        return None

def save_curriculum_state(db: firestore.client, user_id: str, manager: CurriculumManager):
    """Saves current curriculum state to Firestore."""
    if not manager.topic: 
        return
    try:
        doc_ref = get_session_doc_ref(db, user_id, manager.topic)
        doc_ref.set(manager.serialize())
        logger.info("P1 Trace: State saved to Firestore.")
    except Exception as e:
        logger.error("Failed to save state to Firestore: %s", e)

# --- LTM (Long-Term Memory) Helpers ---

def get_ltm_collection_ref(db: firestore.client, user_id: str):
    """Returns the collection reference for quiz attempts."""
    app_id = st.session_state.get('app_id', 'default-app-id')
    ltm_collection_path = f"artifacts/{app_id}/users/{user_id}/quiz_attempts"
    return db.collection(ltm_collection_path)

def log_quiz_attempt(db: firestore.client, user_id: str, node_id: str, is_correct: bool):
    """Logs the result of a quiz attempt to the LTM database."""
    if not db: 
        return
    try:
        ltm_ref = get_ltm_collection_ref(db, user_id)
        ltm_ref.add({
            'timestamp': firestore.SERVER_TIMESTAMP,
            'node_id': node_id,
            'topic': st.session_state.curriculum.topic,
            'status': 'CORRECT' if is_correct else 'INCORRECT'
        })
        logger.info("LTM Trace: Logged attempt for %s: %s", node_id, is_correct)
    except Exception as e:
        logger.error("Failed to log LTM: %s", e)

def get_ltm_history(db: firestore.client, user_id: str, topic: str) -> List[Dict[str, Any]]:
    """Fetches the last N quiz attempts for the topic to inform the Professor Agent."""
    if not db: 
        return []
    try:
        ltm_ref = get_ltm_collection_ref(db, user_id)
        query = ltm_ref.where('topic', '==', topic).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10)
        
        docs = query.stream()
        history = []
        for doc in docs:
            data = doc.to_dict()
            history.append({
                'node': data.get('node_id'),
                'status': data.get('status')
            })
        return history
    except Exception as e:
        logger.error("Failed to fetch LTM history: %s", e)
        return []

# --- SESSION STATE INITIALIZATION ---

if 'curriculum' not in st.session_state:
    st.session_state.curriculum = CurriculumManager()
if 'agent_core' not in st.session_state:
    st.session_state.agent_core = None
if 'db_client' not in st.session_state:
    st.session_state.db_client, st.session_state.user_id = init_firebase()
if 'current_node' not in st.session_state:
    st.session_state.current_node = None
if 'current_content' not in st.session_state:
    st.session_state.current_content = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'initialized_model_name' not in st.session_state:
    st.session_state.initialized_model_name = None
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}

db = st.session_state.db_client
user_id = st.session_state.user_id

# --- UI COMPONENTS ---

def render_sidebar():
    """Renders configuration, metrics, and audit log in the sidebar."""
    
    with st.sidebar:
        st.header("Settings")
        
        with st.container(border=True):
            st.subheader("Agent Configuration")
            st.markdown("This is a **developer demo**. Requires a Gemini API key.")
            api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
            model_name_select = st.selectbox(
                "Model Selection",
                options=[DEFAULT_MODEL, 'gemini-1.5-pro'],
                index=0 if DEFAULT_MODEL == 'gemini-2.5-flash-preview-09-2025' else 1, 
                key="model_name_select", 
                help="Pro offers higher quality content, Flash offers lower latency."
            )

            if st.button("Initialize Agents"):
                if api_key_input:
                    try:
                        st.session_state.agent_core = AgentCore(api_key_input, model_name_select)
                        st.session_state.initialized_model_name = model_name_select
                        st.session_state.history.append("Agents Online (%s)" % model_name_select)
                        st.success("Agents Online")
                    except Exception as e:
                        st.error("Initialization Failed: %s" % e)
                else:
                    st.warning("Please enter your API Key.")

            if st.session_state.agent_core and st.session_state.initialized_model_name != model_name_select:
                 st.warning("Model selection changed. Press 'Initialize Agents' to apply.")
            elif st.session_state.agent_core:
                 st.success("Agents Online (%s)" % st.session_state.initialized_model_name)
            elif not api_key_input:
                 st.warning("Please enter your API Key to initialize Agents.")


        st.markdown("---")
        st.subheader("System Status")
        st.metric("Completed Modules", len(st.session_state.curriculum.completed_nodes))
        st.caption("User ID: %s" % user_id)
        
        # P3: Display Latency Metrics
        if st.session_state.agent_core and st.session_state.agent_core.latency_metrics:
            st.markdown("#### P3: Parallel Agent Tracing")
            for node, latency in st.session_state.agent_core.latency_metrics.items():
                st.metric("%s Latency" % node, "%.2f s" % latency)

        with st.expander("Audit Log"):
            for event in st.session_state.history:
                st.text(event)

def render_initialization():
    """Renders the curriculum creation form."""
    with st.form("init_form"):
        st.subheader("Start New Curriculum")
        topic_input = st.text_input("Enter Learning Goal (e.g., 'Thermodynamics')")
        context_input = st.selectbox("Complexity Level", ["Undergraduate", "Graduate", "PhD"])
        
        load_existing = st.checkbox("Load existing curriculum for this topic?")
        submitted = st.form_submit_button("Start/Generate Curriculum")
        
        if submitted and st.session_state.agent_core:
            topic = topic_input.strip()
            
            if load_existing and db:
                loaded_curriculum = load_curriculum_state(db, user_id, topic)
                if loaded_curriculum:
                    st.session_state.curriculum = loaded_curriculum
                    st.session_state.history.append("Loaded existing session for: %s" % topic)
                    st.success("Loaded existing session for: %s" % topic)
                    st.rerun()
                else:
                    st.warning("No existing session found. Generating new one.")

            if not st.session_state.curriculum.nodes:
                with st.spinner("Architect Agent is designing the dependency graph..."):
                    try:
                        graph_data = st.session_state.agent_core.architect_agent(topic, context_input)
                        if "error" in graph_data:
                            st.error("Generation Error: %s" % graph_data['error'])
                        else:
                            st.session_state.curriculum.load_from_json(graph_data, topic, context_input)
                            st.session_state.history.append("Graph initialized for: %s" % topic)
                            if db:
                                save_curriculum_state(db, user_id, st.session_state.curriculum)
                            st.rerun()
                    except Exception as e:
                        st.error("Agent Failure: %s" % e)

def render_module_view():
    """Renders the main content view with graph, content, and quiz."""
    
    tab_graph, tab_content = st.tabs(["Knowledge Graph", "Current Module"])

    with tab_graph:
        st.subheader("Dependency Structure: %s" % st.session_state.curriculum.topic)
        st.graphviz_chart(st.session_state.curriculum.get_dot_graph())
        
        available_nodes = [
            (nid, data['label']) 
            for nid, data in st.session_state.curriculum.nodes.items() 
            if data['status'] == 'AVAILABLE'
        ]
        
        if available_nodes and st.session_state.current_node is None:
            st.info("Select an available module to begin:")
            selected_node_id = st.selectbox(
                "Available Modules", 
                options=[n[0] for n in available_nodes],
                format_func=lambda x: st.session_state.curriculum.nodes[x]['label']
            )
            
            if st.button("Start Module"):
                st.session_state.current_node = selected_node_id
                st.session_state.quiz_answers = {} 
                
                with st.spinner("Multi-Agent System working: Professor, Proctor, LaTeX, and Verifier..."):
                    node_label = st.session_state.curriculum.nodes[selected_node_id]['label']
                    ltm_history = get_ltm_history(db, user_id, st.session_state.curriculum.topic)
                    content = st.session_state.agent_core.parallel_content_generation(node_label, ltm_history)
                    st.session_state.current_content = content
                    st.session_state.history.append("Started module: %s" % node_label)
                st.rerun()
        elif not available_nodes and not st.session_state.current_node:
            st.success("All required modules completed!")

    with tab_content:
        if st.session_state.current_node:
            node_id = st.session_state.current_node
            node_label = st.session_state.curriculum.nodes[node_id]['label']
            content = st.session_state.current_content
            
            st.header("Module: %s" % node_label)
            
            tab_lecture, tab_quiz = st.tabs(["Lecture Material", "Quiz"])
            
            with tab_lecture:
                # P3: Hallucination Warning Metric (Evaluation)
                audit = content.get('verifier_audit', {})
                risk_score = audit.get('risk_score', 0.0)
                flagged_reason = audit.get('flagged_reason', 'Evaluation pending.')
                
                col_audit_metric, col_audit_reason = st.columns([1, 2])
                
                with col_audit_metric:
                    if risk_score > 0.3:
                        st.error("🚨 Risk Score: %.2f" % risk_score)
                    else:
                        st.success("✅ Confidence: %.2f" % (1.0 - risk_score))
                with col_audit_reason:
                    st.caption("Auditor Note: %s" % flagged_reason)


                st.markdown("---")
                
                # Custom Tool Output (LaTeX Equation)
                latex_data = content.get('latex', {})
                if latex_data.get('latex_equation'):
                    st.subheader("Key Formula")
                    st.latex(latex_data['latex_equation'])
                    st.caption("Relevance: %s" % latex_data.get('reason'))
                    st.markdown("---")
                
                st.markdown(content['lecture'])
            
            with tab_quiz:
                quiz_items = content.get('quiz_items', [])
                num_questions = len(quiz_items)
                
                if num_questions > 0:
                    st.info("Quiz: Answer all %d questions to complete the module. Pass score: %.0f%% (7/10)" % (num_questions, PASS_THRESHOLD * 100))
                    
                    with st.form("quiz_form"):
                        for i, item in enumerate(quiz_items):
                            st.markdown("---")
                            st.markdown("**%d. %s**" % (i + 1, item['question']))
                            
                            user_choice = st.radio(
                                "Select Answer:", 
                                options=item.get('options', ['A', 'B', 'C', 'D']), 
                                index=st.session_state.quiz_answers.get(i, 0),
                                key="q_%d" % i
                            )
                            st.session_state.quiz_answers[i] = item.get('options', ['A', 'B', 'C', 'D']).index(user_choice)

                        submitted = st.form_submit_button("Submit Final Quiz")

                    if submitted:
                        # 1. Calculate Score
                        correct_count = 0
                        for i, item in enumerate(quiz_items):
                            user_selected_index = st.session_state.quiz_answers.get(i, -1) 
                            if user_selected_index == item['correct_option_index']:
                                correct_count += 1
                        
                        score_percentage = correct_count / num_questions
                        is_pass = score_percentage >= PASS_THRESHOLD

                        # 2. A2A Call Setup
                        verifier_audit = content.get('verifier_audit', {})
                        
                        # 3. Trigger Evaluator Agent
                        with st.spinner("Processing results and checking for remediation..."):
                            remedial_plan = st.session_state.agent_core.evaluator_agent(
                                node_label, 
                                score_percentage, 
                                verifier_audit # A2A SIGNAL
                            )
                        
                        # 4. Handle Results (Pass/Fail)
                        if is_pass:
                            st.balloons()
                            st.success("Module Passed! Score: %.0f%% (%d/%d)" % (score_percentage * 100, correct_count, num_questions))
                            log_quiz_attempt(db, user_id, node_id, True)
                            st.session_state.curriculum.mark_completed(node_id)
                        else:
                            st.error("Module Failed. Score: %.0f%% (%d/%d). Needs Remediation." % (score_percentage * 100, correct_count, num_questions))
                            log_quiz_attempt(db, user_id, node_id, False)
                            
                            if remedial_plan and not remedial_plan.get('error'):
                                injected = st.session_state.curriculum.inject_remedial_node(
                                    node_id, 
                                    remedial_plan
                                )
                                if injected:
                                    st.warning("Curriculum Updated: Added remedial node '%s'" % remedial_plan['remedial_node_label'])
                                    st.session_state.history.append("Remediation injected: %s" % remedial_plan['remedial_node_label'])

                        # 5. Cleanup and Rerun
                        if db:
                            save_curriculum_state(db, user_id, st.session_state.curriculum)
                        st.session_state.current_node = None
                        st.session_state.current_content = None
                        st.session_state.quiz_answers = {}
                        time.sleep(3)
                        st.rerun()

                else:
                    st.warning("No quiz questions generated for this module.")

# --- MAIN APP EXECUTION ---

st.set_page_config(layout="wide")

st.title("Dynamic Curriculum Graph Generator")
st.markdown("A multi-agent higher education system featuring: **A2A Protocol, Parallel Agents, LTM, and Custom Tools**.")

render_sidebar()

if not st.session_state.curriculum.nodes:
    if st.session_state.agent_core:
        render_initialization()
    else:
        st.error("Agents not initialized. Please configure settings in the sidebar.")
else:
    render_module_view()