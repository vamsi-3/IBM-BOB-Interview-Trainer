# IBM-BOB-Interview-Trainer
Interview guidance for technical, behavioral, HR, evaluation, and experience-level interview training. Used to generate role-appropriate questions, evaluate candidate answers, and provide actionable feedback.
# IBM BOB Problem Statement #22: AI Interview Trainer Agent

An AI-powered interview trainer built on **IBM watsonx Orchestrate** and **IBM Cloud** that conducts personalized, interactive mock interviews for candidates across technical, behavioral, and HR domains.

## 🚀 Key Features
- **Personalized Onboarding**: Tailors question domain and difficulty based on candidate experience, role, and key skills.
- **Knowledge Base Integration**: Retrieves real-world technical and behavioral questions and model answers from study materials.
- **Structured Feedback & Scoring**: Evaluates candidate responses using Markdown scoring tables ($1-10$ scale), identifying strengths, missing gaps, and actionable improvement tips.
- **Interactive Session Flow**: One-question-at-a-time flow with support for interactive commands (`skip`, `quit`).

## 🛠️ Architecture & Tech Stack
- **Platform**: IBM watsonx Orchestrate
- **LLM Model**: Watsonx Orchestrate Frontier
- **Knowledge Base**: Domain-specific technical and behavioral interview RAG knowledge base
- **Core Engine**: Python 3.12 (Candidate Profile, Interview Engine, Prompt Templates)

## 📁 Repository Structure
- `candidate_profile.py`: Manages candidate onboarding data and interview state.
- `interview_engine.py`: Orchestrates question selection, scoring, and feedback logic.
- `prompt_templates.py`: Contains evaluation guidelines and onboarding prompt specifications.
- `watsonx_client.py`: API interface for watsonx services.
- `knowledge_base/`: RAG knowledge definitions (`rag_knowledge.py`).
- `workspace_config.yaml`: Environment configuration.
