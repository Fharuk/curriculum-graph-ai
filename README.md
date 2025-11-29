# curriculum-graph-ai

Dynamic Curriculum Graph Generator (Capstone Project)

Project Summary

The Dynamic Curriculum Graph Generator is a high-performance, multi-agent system designed to address the static nature of traditional syllabi in Higher Education. It leverages the Gemini API, Firebase Firestore, and advanced agent orchestration patterns to deliver truly personalized and adaptive learning pathways.

This project serves as a capstone demonstration of expertise in integrating multiple AI design principles: Parallel Execution, Sequential Feedback, Long-Term Memory (LTM), and the Agent-to-Agent (A2A) Protocol.

Core Architectural Features

The system is built on a robust multi-agent framework to ensure speed, intelligence, and reliability.

1. Multi-Agent Orchestration (P3)

Parallel Agents (Speed): The Professor (Content), Proctor (Quiz), and LaTeX (Custom Tool) Agents run concurrently to minimize latency, proven by the Parallel Agent Tracing Metric displayed in the sidebar.

Sequential Agents (Audit): The Verifier Agent runs immediately after content generation to check for factual confidence before the material is presented to the student.

2. A2A Protocol and Evaluation

A2A Implementation: The Verifier Agent passes its Hallucination Risk Score (the A2A signal) directly to the Evaluator Agent.

Decision-Making: If the risk score is above 0.5, the Evaluator Agent modifies the remedial node's label to include a [CONTENT WARNING], demonstrating autonomous, structured communication between two agents to safeguard academic integrity.

Academic Rigor: Quizzes are set at 10 questions with a passing threshold of 70%.

3. Sessions & Memory Management (P1/LTM)

Persistence (P1): The entire curriculum structure (nodes, edges, completion status) is saved in Firebase Firestore, allowing the user to resume their session across different devices or browser sessions (Long-Running Operations).

Long-Term Memory (LTM): The Professor Agent fetches the user's last 10 quiz attempts from the database before writing a lecture. This forces the content to be tailored to the student's specific failure patterns, ensuring true personalization.

4. Custom Tools

LaTeX Generator: The system includes a Custom Tool (LaTeX Agent) that generates and renders relevant mathematical or scientific equations directly within the lecture material, essential for STEM education.

Technical Requirements

Prerequisites

Python: 3.10 or higher

Gemini API Key: Required to run the agent architecture.

GitHub Repository: (Publicly accessible for Streamlit Cloud deployment)

Deployment Dependencies

The project relies on two files to configure the Streamlit container:

requirements.txt: Specifies Python packages.

packages.txt: Specifies system-level packages (essential for Graphviz).

Setup and Local Execution

To run the project locally (or within a GitHub Codespace):

Step 1: Clone the Repository and Install System Dependencies

# Clone the repository
git clone [https://github.com/your-username/curriculum-graph-ai.git](https://github.com/fharuk/curriculum-graph-ai.git)
cd curriculum-graph-ai

# Install Graphviz system package (Mandatory for visualization)
sudo apt-get update && sudo apt-get install -y graphviz


Step 2: Install Python Dependencies

pip install -r requirements.txt


Step 3: Run the Application

The application must be run using the secure network configuration flags.

streamlit run app.py --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
