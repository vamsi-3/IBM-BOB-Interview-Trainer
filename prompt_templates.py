"""
Prompt Templates for Interview Trainer Agent
---------------------------------------------
All prompts are assembled here to keep the agent logic clean.
"""

from candidate_profile import CandidateProfile


# ── System Role ──────────────────────────────────────────────────────────────

SYSTEM_ROLE = (
    "You are an expert AI Interview Trainer. "
    "Your job is to help candidates prepare for job interviews by asking "
    "relevant, personalised interview questions, evaluating answers thoroughly, "
    "and providing constructive, professional feedback. "
    "You always keep questions appropriate to the candidate's experience level and target role. "
    "You ask one question at a time and wait for the candidate's answer. "
    "You never invent facts about a candidate's background. "
    "You ground your questions and feedback in retrieved knowledge when available."
)


# ── Question Generation ───────────────────────────────────────────────────────

def build_question_prompt(
    profile: CandidateProfile,
    rag_context: str,
    question_number: int,
    total_questions: int,
    previous_questions: list,
    interview_type: str,
) -> str:
    """
    Build the prompt to generate the next interview question.

    Parameters
    ----------
    profile : CandidateProfile
    rag_context : str
        Retrieved knowledge context.
    question_number : int
        1-based current question index.
    total_questions : int
        Total questions planned.
    previous_questions : list
        List of already-asked question strings (to avoid repeats).
    interview_type : str
        'technical' | 'behavioral' | 'hr' | 'mixed'
    """
    prev_q_text = ""
    if previous_questions:
        formatted = "\n".join(f"  Q{i+1}: {q}" for i, q in enumerate(previous_questions))
        prev_q_text = f"\nPreviously asked questions (do NOT repeat these):\n{formatted}\n"

    jd_section = ""
    if profile.job_description.strip():
        jd_section = f"\nJob Description provided by candidate:\n{profile.job_description[:1500]}\n"

    resume_section = ""
    if profile.resume_text.strip():
        resume_section = f"\nCandidate Resume / Background:\n{profile.resume_text[:2000]}\n"

    type_instruction = {
        "technical": (
            "Ask a TECHNICAL question relevant to the candidate's skills and target role. "
            "This may involve coding concepts, system design, algorithms, or domain knowledge."
        ),
        "behavioral": (
            "Ask a BEHAVIOURAL question using a scenario or situation-based format. "
            "Focus on past experiences, teamwork, conflict resolution, or problem-solving."
        ),
        "hr": (
            "Ask an HR question about motivation, career goals, strengths/weaknesses, "
            "or cultural fit."
        ),
        "mixed": (
            f"This is question {question_number} of {total_questions}. "
            "Vary the question type: use a mix of technical, behavioural, and HR questions "
            "across the full interview. "
            "For questions 1–2 ask technical, questions 3–4 ask behavioural, "
            "question 5+ ask HR or mixed."
        ),
    }.get(interview_type.lower(), "Ask a relevant interview question.")

    prompt = f"""{SYSTEM_ROLE}

=== CANDIDATE PROFILE ===
{profile.summary()}
{resume_section}{jd_section}
=== RETRIEVED KNOWLEDGE CONTEXT ===
{rag_context}

=== TASK ===
You are conducting question {question_number} of {total_questions} in this interview.

{type_instruction}

Requirements:
- The question must be directly relevant to the candidate's target role ({profile.target_role}) and skills ({', '.join(profile.skills[:5])}).
- Keep the difficulty appropriate for a {profile.experience_level} candidate.
- Ask exactly ONE question only. Do not ask multiple questions at once.
- Do not provide the answer or hints.
- Do not repeat any previously asked question.
- Format: Just the question itself, no preamble like "Here is a question:".
{prev_q_text}
Interview Question {question_number}:"""

    return prompt


# ── Answer Evaluation ─────────────────────────────────────────────────────────

def build_evaluation_prompt(
    profile: CandidateProfile,
    question: str,
    candidate_answer: str,
    rag_context: str,
    question_type: str,
) -> str:
    """
    Build the prompt to evaluate a candidate's answer.

    Returns a structured evaluation with score, strengths, weaknesses,
    improvement advice, and a model answer.
    """
    jd_section = ""
    if profile.job_description.strip():
        jd_section = f"\nJob Description:\n{profile.job_description[:1000]}\n"

    prompt = f"""{SYSTEM_ROLE}

=== CANDIDATE PROFILE ===
{profile.summary()}
{jd_section}
=== RETRIEVED KNOWLEDGE CONTEXT ===
{rag_context}

=== EVALUATION TASK ===
Interview Question ({question_type}):
{question}

Candidate's Answer:
{candidate_answer}

Evaluate the candidate's answer thoroughly and professionally.
The candidate is a {profile.experience_level} targeting the role of {profile.target_role}.

Provide your evaluation in EXACTLY this format:

SCORE: [X/10]

STRENGTHS:
- [List what the candidate did well — be specific]
- [Add more bullet points as needed]

WEAKNESSES / MISSING POINTS:
- [List what was missing, incorrect, or could be improved — be specific]
- [Add more bullet points as needed]

IMPROVEMENT ADVICE:
[Give specific, actionable advice to improve this answer. Be constructive and encouraging.]

MODEL ANSWER:
[Provide a strong example answer that demonstrates what an excellent response looks like for a {profile.experience_level} at this role. Ground it in the retrieved knowledge where relevant.]

---
Evaluation:"""

    return prompt


# ── Final Report ──────────────────────────────────────────────────────────────

def build_final_report_prompt(
    profile: CandidateProfile,
    qa_history: list,
    rag_context: str,
) -> str:
    """
    Build the prompt to generate the final interview performance report.

    Parameters
    ----------
    qa_history : list
        List of dicts: [{"question": ..., "answer": ..., "evaluation": ..., "score": ...}]
    """
    history_text = ""
    for i, item in enumerate(qa_history, 1):
        history_text += (
            f"\nQ{i} [{item.get('type', 'unknown')}]: {item['question']}\n"
            f"Candidate Answer: {item['answer']}\n"
            f"Evaluation Summary: {item['evaluation'][:500]}...\n"
            f"Score: {item.get('score', 'N/A')}/10\n"
        )

    scores = [item.get("score", 0) for item in qa_history if isinstance(item.get("score"), (int, float))]
    avg_score = round(sum(scores) / len(scores), 1) if scores else "N/A"

    prompt = f"""{SYSTEM_ROLE}

=== CANDIDATE PROFILE ===
{profile.summary()}

=== INTERVIEW HISTORY ===
{history_text}

=== RETRIEVED KNOWLEDGE CONTEXT ===
{rag_context}

=== TASK ===
The interview for {profile.name} targeting the role of {profile.target_role} is now complete.
Their average score across {len(qa_history)} questions was approximately {avg_score}/10.

Generate a comprehensive, professional FINAL INTERVIEW REPORT in EXACTLY this format:

FINAL INTERVIEW REPORT
======================
Candidate       : {profile.name}
Target Role     : {profile.target_role}
Experience Level: {profile.experience_level}
Interview Type  : {profile.interview_type}
Questions Asked : {len(qa_history)}
Average Score   : {avg_score}/10

OVERALL PERFORMANCE SUMMARY:
[2–3 sentences summarising overall performance]

TECHNICAL PERFORMANCE:
[Assessment of technical knowledge, accuracy, and depth shown during the interview]

COMMUNICATION PERFORMANCE:
[Assessment of clarity, structure, and ability to articulate ideas]

BEHAVIOURAL PERFORMANCE:
[Assessment of STAR usage, examples quality, and interpersonal skills demonstrated]

KEY STRENGTHS:
- [Bullet list of the candidate's strongest demonstrated qualities]

KEY WEAKNESSES:
- [Bullet list of areas needing improvement]

TOPICS REQUIRING IMPROVEMENT:
- [Specific technical or behavioural topics to study before the actual interview]

RECOMMENDED PREPARATION PLAN:
[A practical 5–7 day preparation plan with specific actions, resources, and focus areas tailored to this candidate's gaps]

FINAL VERDICT:
[One of: "Strong candidate — ready for interviews", "Good candidate — minor preparation needed", 
"Moderate candidate — focused preparation required", "Needs significant preparation before interviewing"]

---
Report:"""

    return prompt
