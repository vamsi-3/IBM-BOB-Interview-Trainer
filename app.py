"""
Flask Web Interface for AI Interview Trainer Agent
===================================================
Wraps the existing Python CLI engine in a REST API so the
interview can be driven from a browser instead of a terminal.

Run:
    python app.py

Then open:  http://localhost:5000
"""

import os
import uuid
import json
import threading
from flask import Flask, request, jsonify, render_template, session

from watsonx_client import WatsonXClient
from candidate_profile import CandidateProfile
from interview_engine import InterviewEngine, QUESTION_COUNT, _detect_question_type, _extract_score
from knowledge_base.rag_knowledge import retrieve_for_profile
from prompt_templates import (
    build_question_prompt,
    build_evaluation_prompt,
    build_final_report_prompt,
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ── In-memory session store ───────────────────────────────────────────────────
# Keyed by session_id: stores profile, rag_context, qa_history, llm_client, etc.
_sessions: dict = {}
_sessions_lock = threading.Lock()


def _get_session(sid: str) -> dict:
    with _sessions_lock:
        return _sessions.get(sid)


def _put_session(sid: str, data: dict):
    with _sessions_lock:
        _sessions[sid] = data


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_interview():
    """
    Body (JSON):
    {
        "api_key": "...",
        "name": "...",
        "experience_level": "fresher|junior|mid-level|senior|lead/principal",
        "years_of_experience": 3,        // optional, int
        "target_role": "Data Scientist",
        "skills": ["Python", "SQL"],
        "interview_type": "technical|behavioral|hr|mixed",
        "resume_text": "...",
        "job_description": "..."         // optional
    }
    """
    body = request.get_json(force=True)

    # Validate required fields
    required = ["api_key", "name", "experience_level", "target_role", "skills", "interview_type", "resume_text"]
    for field in required:
        if not body.get(field):
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # Connect to WatsonX
    try:
        client = WatsonXClient(api_key=body["api_key"])
        client._ensure_token()
    except Exception as exc:
        return jsonify({"error": f"WatsonX connection failed: {exc}"}), 401

    skills = body["skills"]
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]

    years = body.get("years_of_experience")
    if years is not None:
        try:
            years = int(years)
        except (ValueError, TypeError):
            years = None

    profile = CandidateProfile(
        name=body["name"],
        experience_level=body["experience_level"],
        target_role=body["target_role"],
        skills=skills,
        resume_text=body["resume_text"],
        job_description=body.get("job_description", ""),
        interview_type=body["interview_type"],
        years_of_experience=years,
    )

    # RAG retrieval
    rag_context, rag_docs = retrieve_for_profile(
        role=profile.target_role,
        experience_level=profile.experience_level,
        skills=profile.skills,
        interview_type=profile.interview_type,
        top_k=6,
    )

    total_questions = QUESTION_COUNT.get(profile.interview_type.lower(), 5)

    sid = str(uuid.uuid4())
    _put_session(sid, {
        "profile": profile,
        "rag_context": rag_context,
        "rag_docs": rag_docs,
        "qa_history": [],
        "client": client,
        "total_questions": total_questions,
        "current_question_num": 0,
        "current_question": None,
        "current_q_type": None,
    })

    return jsonify({
        "session_id": sid,
        "total_questions": total_questions,
        "rag_docs_retrieved": len(rag_docs),
        "profile_summary": profile.summary(),
    })


@app.route("/api/question", methods=["POST"])
def get_question():
    """
    Body: { "session_id": "..." }
    Returns the next interview question.
    """
    body = request.get_json(force=True)
    sid = body.get("session_id")
    sess = _get_session(sid)
    if not sess:
        return jsonify({"error": "Session not found or expired"}), 404

    profile = sess["profile"]
    q_num = sess["current_question_num"] + 1

    if q_num > sess["total_questions"]:
        return jsonify({"error": "No more questions", "done": True}), 200

    q_type = _detect_question_type(q_num, profile.interview_type, sess["total_questions"])
    previous_questions = [item["question"] for item in sess["qa_history"]]

    prompt = build_question_prompt(
        profile=profile,
        rag_context=sess["rag_context"],
        question_number=q_num,
        total_questions=sess["total_questions"],
        previous_questions=previous_questions,
        interview_type=q_type,
    )

    try:
        question = sess["client"].generate(prompt, max_new_tokens=300, temperature=0.7)
    except Exception as exc:
        return jsonify({"error": f"Question generation failed: {exc}"}), 500

    sess["current_question_num"] = q_num
    sess["current_question"] = question
    sess["current_q_type"] = q_type
    _put_session(sid, sess)

    return jsonify({
        "question_number": q_num,
        "total_questions": sess["total_questions"],
        "question_type": q_type,
        "question": question,
    })


@app.route("/api/answer", methods=["POST"])
def submit_answer():
    """
    Body: { "session_id": "...", "answer": "..." }
    Returns evaluation and score.
    """
    body = request.get_json(force=True)
    sid = body.get("session_id")
    sess = _get_session(sid)
    if not sess:
        return jsonify({"error": "Session not found or expired"}), 404

    answer = (body.get("answer") or "").strip()
    question = sess["current_question"]
    q_type = sess["current_q_type"]
    profile = sess["profile"]

    if not answer or answer.lower() == "skip":
        sess["qa_history"].append({
            "question": question,
            "answer": "[Skipped]",
            "evaluation": "Question was skipped.",
            "score": 0,
            "type": q_type,
        })
        _put_session(sid, sess)
        return jsonify({
            "skipped": True,
            "question_number": sess["current_question_num"],
            "total_questions": sess["total_questions"],
            "done": sess["current_question_num"] >= sess["total_questions"],
        })

    prompt = build_evaluation_prompt(
        profile=profile,
        question=question,
        candidate_answer=answer,
        rag_context=sess["rag_context"],
        question_type=q_type,
    )

    try:
        evaluation = sess["client"].generate(prompt, max_new_tokens=900, temperature=0.5)
    except Exception as exc:
        evaluation = f"Evaluation unavailable: {exc}"

    score = _extract_score(evaluation)

    sess["qa_history"].append({
        "question": question,
        "answer": answer,
        "evaluation": evaluation,
        "score": score,
        "type": q_type,
    })
    _put_session(sid, sess)

    done = sess["current_question_num"] >= sess["total_questions"]

    return jsonify({
        "evaluation": evaluation,
        "score": score,
        "question_number": sess["current_question_num"],
        "total_questions": sess["total_questions"],
        "done": done,
    })


@app.route("/api/report", methods=["POST"])
def get_report():
    """
    Body: { "session_id": "..." }
    Generates and returns the final interview report.
    """
    body = request.get_json(force=True)
    sid = body.get("session_id")
    sess = _get_session(sid)
    if not sess:
        return jsonify({"error": "Session not found or expired"}), 404

    profile = sess["profile"]
    qa_history = sess["qa_history"]

    if not qa_history:
        return jsonify({"error": "No answers recorded — nothing to report"}), 400

    prompt = build_final_report_prompt(
        profile=profile,
        qa_history=qa_history,
        rag_context=sess["rag_context"],
    )

    try:
        report = sess["client"].generate(prompt, max_new_tokens=1200, temperature=0.4)
    except Exception as exc:
        return jsonify({"error": f"Report generation failed: {exc}"}), 500

    scores = [
        item["score"] for item in qa_history
        if isinstance(item.get("score"), (int, float))
    ]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    return jsonify({
        "report": report,
        "avg_score": avg_score,
        "questions_answered": len(qa_history),
        "qa_history": [
            {
                "question_number": i + 1,
                "type": item["type"],
                "question": item["question"],
                "answer": item["answer"],
                "score": item.get("score"),
            }
            for i, item in enumerate(qa_history)
        ],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  AI Interview Trainer — Web UI")
    print(f"  Open your browser at: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
