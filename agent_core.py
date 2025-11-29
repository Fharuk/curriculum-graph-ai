import json
import logging
import concurrent.futures
import time
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class AgentCore:
    """
    Orchestrates Gemini API interactions using multi-agent patterns (Parallel and Sequential).
    Implements Custom Tools (LaTeX Agent), Agent Evaluation (Verifier Agent), and A2A Protocol.
    """
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("API Key is required.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name) 
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.latency_metrics: Dict[str, float] = {}

    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        """Handles JSON generation and robust output sanitization."""
        
        tools = None 
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
                tools=tools
            )
            raw_data = json.loads(response.text)
            
            # Sanitization: Handle cases where model wraps JSON object in a list
            if isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], dict):
                raw_data = raw_data[0]
            
            if isinstance(raw_data, dict):
                return raw_data
            
            logger.error("LLM returned unexpected type: %s", type(raw_data))
            return {"error": "LLM returned non-dictionary structure."}

        except json.JSONDecodeError as e:
            logger.error("JSON Decoding Failed: %s", e)
            return {"error": "LLM returned invalid JSON: %s..." % response.text[:100]}
        except Exception as e:
            logger.error("LLM Generation Failed (Unknown Error): %s", e)
            return {"error": str(e)}

    # --- AGENT 1: CURRICULUM ARCHITECT ---
    def architect_agent(self, topic: str, user_context: str) -> Dict[str, Any]:
        """Generates the initial dependency graph structure."""
        logger.info("Architect Agent activated for topic: %s", topic)
        
        prompt = f"""
        You are a Curriculum Architect for Higher Education. Based on the topic, determine the core 
        prerequisite structure.
        Topic: {topic}
        Context: {user_context}

        Task: Create a directed acyclic graph (DAG) of learning concepts for this topic.
        
        Requirements:
        1. 'nodes': A list of concepts. Each must have a unique 'id' (string) and a 'label' (string).
        2. 'edges': A list of dependencies where 'source' is the prerequisite and 'target' is the advanced concept.
        3. Keep it between 5-8 nodes for a manageable session.
        4. Ensure the graph starts with foundational concepts.

        Output JSON format:
        {{
            "nodes": [{{"id": "c1", "label": "Concept Name"}}],
            "edges": [{{"source": "c1", "target": "c2"}}]
        }}
        """
        return self._generate_json(prompt)

    # --- AGENT 2: CUSTOM TOOL - LATEX GENERATOR ---
    def _latex_agent_task(self, node_label: str) -> Dict[str, str]:
        """Generates a relevant LaTeX equation or formula."""
        prompt = f"""
        You are a Formula Generator. For the concept '{node_label}', identify the most important 
        mathematical or scientific equation. If one is highly relevant, provide it in LaTeX.
        If not relevant, return empty.
        
        Output JSON: {{"latex_equation": "Your LaTeX code here (e.g., \\frac{{d}}{{dx}} x^2 = 2x )", "reason": "why this formula is relevant"}}
        """
        return self._generate_json(prompt)

    # --- AGENT 3: EVALUATION - VERIFIER AGENT (Sequential) ---
    def _verifier_agent_task(self, lecture_content: str) -> Dict[str, Any]:
        """Sequential Agent that checks confidence (Hallucination Warning Metric)."""
        prompt = f"""
        You are a Content Auditor. Evaluate the academic confidence and factual certainty of the 
        following lecture material.

        Lecture: {lecture_content}

        Task: Assign a Hallucination Risk Score from 0.0 (No Risk/High Confidence) to 1.0 (High Risk/Low Confidence).
        
        Output JSON: 
        {{
            "risk_score": 0.0 (float),
            "flagged_reason": "Briefly state why the score is above 0.3, or 'Content is highly factual' otherwise."
        }}
        """
        return self._generate_json(prompt)

    # --- AGENT 4: PROFESSOR AGENT (LTM CONSUMER) ---
    def _content_agent_task(self, node_label: str, ltm_history: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generates lecture content, tailoring it based on the user's LTM.
        """
        history_summary = "No significant prior history found."
        if ltm_history:
            failed_nodes = [h['node'] for h in ltm_history if h['status'] == 'INCORRECT']
            success_count = len([h for h in ltm_history if h['status'] == 'CORRECT'])
            
            if failed_nodes:
                history_summary = f"Student has recently struggled/failed on related concepts: {', '.join(set(failed_nodes))}. Focus the explanation on foundational gaps."
            elif success_count >= 3:
                 history_summary = "Student has a strong track record. Ensure the explanation is concise and moves quickly to application."

        prompt = f"""
        You are a Professor. Your goal is to write a tailored lecture.

        LTM CONTEXT (MUST USE THIS): {history_summary}

        Explain the concept '{node_label}' clearly and concisely (approx 300 words).
        Focus on academic rigor suitable for undergraduates. Integrate a reference to a key formula.
        
        If the LTM CONTEXT indicates struggle, use more detailed examples and analogies.
        
        Output JSON: {{"content_text": "YOUR_EXPLANATION"}}
        """
        return self._generate_json(prompt)

    # --- AGENT 5: PROCTOR AGENT (10-Question Quiz) ---
    def _quiz_agent_task(self, node_label: str) -> Dict[str, Any]:
        """Generates the quiz questions and answer key (10 questions)."""
        prompt = f"""
        You are a Proctor. Create ten (10) distinct multiple-choice questions to test deep understanding of '{node_label}'.
        Each question must have four options (A, B, C, D) and one correct answer index.

        Output JSON must be a single object containing a list of 10 quiz items:
        {{
            "quiz_items": [
                {{"question": "Q1 text", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_option_index": 0, "explanation": "Why correct"}},
                // ... 9 more items ...
            ]
        }}
        """
        return self._generate_json(prompt)

    # --- PARALLEL EXECUTION FLOW (Core Capstone Demonstration) ---
    def parallel_content_generation(self, node_label: str, ltm_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes Professor, Proctor, and LaTeX agents in parallel, followed by the Verifier Agent sequentially.
        """
        logger.info("Starting parallel execution for node: %s", node_label)
        start_time = time.time()

        # Parallel Tasks: Content, Quiz, and Custom Tool (LaTeX)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_content = executor.submit(self._content_agent_task, node_label, ltm_history)
            future_quiz = executor.submit(self._quiz_agent_task, node_label)
            future_latex = executor.submit(self._latex_agent_task, node_label)
            
            content_result = future_content.result()
            quiz_result = future_quiz.result()
            latex_result = future_latex.result()

        # Sequential Task: Verifier Agent (Agent Evaluation)
        verifier_result = self._verifier_agent_task(content_result.get("content_text", ""))
        
        latency = time.time() - start_time
        
        # P3: Store Observability metric
        self.latency_metrics[node_label] = latency
        logger.info("P3 Trace: Multi-step content generation for %s took %.2f seconds.", node_label, latency)

        return {
            "lecture": content_result.get("content_text", "Error generating content."),
            "quiz_items": quiz_result.get("quiz_items", []),
            "latex": latex_result,
            "verifier_audit": verifier_result
        }

    # --- AGENT 6: EVALUATOR AGENT (A2A Protocol Consumer) ---
    def evaluator_agent(self, node_label: str, user_score_percentage: float, verifier_audit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Decides on remedial action after quiz failure. Consumes A2A signal from Verifier Agent.
        """
        # Score check is now handled by the caller (app.py) based on 70% threshold
        if user_score_percentage >= 0.70:
            return None

        logger.info("Evaluator Agent triggered for failure on: %s", node_label)
        
        # A2A Protocol Implementation
        risk_score = verifier_audit.get('risk_score', 0.0)
        
        remedial_prefix = ""
        if risk_score > 0.5:
            # A2A Signal: Verifier Agent (Audit) flags high risk (A2A)
            remedial_prefix = "[CONTENT WARNING] Review source material for: "
            logger.warning("A2A Protocol: Verifier flagged high risk (%.2f). Modifying remediation.", risk_score)

        prompt = f"""
        The student scored {user_score_percentage*100:.0f}% on the quiz for '{node_label}', indicating a knowledge gap.
        Identify a SPECIFIC single missing prerequisite concept that explains this failure.
        
        Output JSON:
        {{
            "remedial_node_id": "remedial_{node_label.replace(' ', '_')}",
            "remedial_node_label": "Name of the sub-concept",
            "reason": "Brief reason why this is needed."
        }}
        """
        
        remedial_plan = self._generate_json(prompt)
        
        # Apply A2A modification to the label
        if 'remedial_node_label' in remedial_plan:
            remedial_plan['remedial_node_label'] = remedial_prefix + remedial_plan['remedial_node_label']

        return remedial_plan