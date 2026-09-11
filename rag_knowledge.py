"""
RAG Knowledge Base for Interview Trainer Agent
-----------------------------------------------
Lightweight in-memory knowledge store using TF-IDF style retrieval.
No external vector database required — suitable for MVP demo.
"""

import re
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Knowledge Documents
# ---------------------------------------------------------------------------

KNOWLEDGE_DOCUMENTS = [
    # ── TECHNICAL INTERVIEW GUIDANCE ────────────────────────────────────────
    {
        "id": "tech_001",
        "category": "technical",
        "roles": ["software engineer", "developer", "programmer", "backend", "frontend", "fullstack", "full stack"],
        "title": "Data Structures & Algorithms Interview Expectations",
        "content": (
            "Technical interviews for software engineers typically assess data structures (arrays, linked lists, "
            "trees, graphs, hash maps, heaps) and algorithms (sorting, searching, dynamic programming, recursion, "
            "greedy algorithms). Candidates should be able to discuss time and space complexity using Big-O notation. "
            "Strong answers include: stating the problem understanding, choosing an appropriate data structure, "
            "walking through an example, coding the solution clearly, and analysing complexity. "
            "Example question: 'Find the two numbers in an array that add up to a target sum.' "
            "Strong answer: Use a hash map for O(n) time complexity — iterate once, storing complement lookups."
        ),
    },
    {
        "id": "tech_002",
        "category": "technical",
        "roles": ["software engineer", "developer", "backend", "fullstack"],
        "title": "System Design Interview Expectations",
        "content": (
            "System design interviews evaluate the ability to architect scalable, reliable systems. "
            "Key areas: load balancing, caching (Redis, Memcached), databases (SQL vs NoSQL), "
            "message queues (Kafka, RabbitMQ), microservices, API design (REST, GraphQL), "
            "CDNs, horizontal vs vertical scaling, CAP theorem. "
            "Strong answers clarify requirements first, estimate scale, design components incrementally, "
            "explain trade-offs, and handle failure scenarios. "
            "Example: 'Design a URL shortener.' Discuss storage estimation, hashing strategy, "
            "database choice, read-heavy caching, and redirect flow."
        ),
    },
    {
        "id": "tech_003",
        "category": "technical",
        "roles": ["data scientist", "data analyst", "ml engineer", "machine learning", "ai engineer"],
        "title": "Data Science & ML Interview Expectations",
        "content": (
            "Data science interviews cover statistics, probability, machine learning fundamentals, "
            "and practical ML workflow. Topics: bias-variance tradeoff, overfitting/underfitting, "
            "cross-validation, feature engineering, supervised vs unsupervised learning, "
            "gradient descent, regularisation (L1/L2), evaluation metrics (precision, recall, F1, AUC-ROC). "
            "Candidates should explain model selection rationale and handle imbalanced datasets. "
            "Strong answer example for 'What is the bias-variance tradeoff?': "
            "High bias = underfitting (model too simple); high variance = overfitting (model too complex). "
            "Goal is the sweet spot that minimises total error on unseen data."
        ),
    },
    {
        "id": "tech_004",
        "category": "technical",
        "roles": ["devops", "sre", "cloud engineer", "infrastructure"],
        "title": "DevOps & Cloud Interview Expectations",
        "content": (
            "DevOps interviews assess CI/CD pipelines, containerisation (Docker, Kubernetes), "
            "infrastructure-as-code (Terraform, Ansible), monitoring (Prometheus, Grafana), "
            "cloud platforms (AWS, GCP, Azure, IBM Cloud), and SRE principles (SLOs, SLAs, error budgets). "
            "Strong candidates explain deployment strategies (blue-green, canary), "
            "incident response processes, and automation philosophy. "
            "Example: 'Describe your CI/CD pipeline.' Strong answer covers: source control triggers, "
            "automated testing stages, Docker image builds, Kubernetes rollouts, and rollback mechanisms."
        ),
    },
    {
        "id": "tech_005",
        "category": "technical",
        "roles": ["product manager", "product owner", "pm"],
        "title": "Product Manager Technical Expectations",
        "content": (
            "Product manager interviews combine product sense, analytical thinking, and execution ability. "
            "Areas: defining product vision, writing PRDs, prioritisation frameworks (RICE, MoSCoW), "
            "OKRs, A/B testing, metrics (DAU, MAU, retention, NPS, conversion), "
            "stakeholder management, and roadmap planning. "
            "Strong candidates use structured frameworks: situation → insight → decision → outcome. "
            "Example: 'How would you improve our onboarding flow?' "
            "Strong answer: Define success metric, identify drop-off points via funnel analysis, "
            "hypothesis-driven experiments, prioritise highest-impact changes, measure results."
        ),
    },

    # ── BEHAVIOURAL INTERVIEW GUIDANCE ──────────────────────────────────────
    {
        "id": "behav_001",
        "category": "behavioral",
        "roles": ["all"],
        "title": "STAR Method for Behavioural Questions",
        "content": (
            "Behavioural interview questions use the STAR framework: Situation, Task, Action, Result. "
            "Situation: set the context briefly. Task: describe your specific responsibility. "
            "Action: explain the concrete steps YOU took (use 'I', not 'we'). "
            "Result: quantify the outcome where possible. "
            "Common questions: 'Tell me about a time you handled conflict', "
            "'Describe a project where you failed and what you learned', "
            "'Give an example of leading without authority', "
            "'Tell me about a time you managed competing priorities'. "
            "Strong answers are specific, concise (2–3 minutes), honest about challenges, "
            "and demonstrate self-awareness and growth."
        ),
    },
    {
        "id": "behav_002",
        "category": "behavioral",
        "roles": ["all"],
        "title": "Leadership & Teamwork Behavioural Questions",
        "content": (
            "Leadership questions assess initiative, influence, and collaboration. "
            "Interviewers look for: ownership mindset, ability to motivate peers, "
            "conflict resolution skills, and results-orientation. "
            "Example: 'Describe a time you led a project under tight deadlines.' "
            "Strong answer elements: clear goal-setting, delegation, unblocking impediments, "
            "transparent communication with stakeholders, successful delivery with measurable outcome. "
            "Avoid generic answers. Be specific about YOUR actions and the tangible impact."
        ),
    },
    {
        "id": "behav_003",
        "category": "behavioral",
        "roles": ["all"],
        "title": "Problem-Solving & Adaptability Questions",
        "content": (
            "Problem-solving questions test analytical thinking, creativity, and resilience. "
            "Common questions: 'Tell me about a time requirements changed mid-project', "
            "'How did you resolve a technical disagreement with a colleague?', "
            "'Describe solving a problem with incomplete information.' "
            "Strong answers demonstrate: structured thinking, data-driven decisions, "
            "willingness to pivot, and a learning mindset. "
            "Evaluators reward candidates who show they proactively identified the problem "
            "rather than waiting to be told, and who reflect on what they would do differently."
        ),
    },

    # ── HR INTERVIEW GUIDANCE ────────────────────────────────────────────────
    {
        "id": "hr_001",
        "category": "hr",
        "roles": ["all"],
        "title": "Common HR Interview Questions & Strong Answers",
        "content": (
            "HR interviews assess cultural fit, motivation, and communication. "
            "Key questions and strong answer strategies:\n"
            "1. 'Tell me about yourself.' — 2-minute professional summary: background, key experiences, "
            "skills, and why you're excited about this role. Not a biography.\n"
            "2. 'Why do you want this role?' — Show genuine interest; connect your strengths to the role's "
            "needs; reference specific aspects of the company/team.\n"
            "3. 'What are your strengths?' — Pick 2–3 relevant strengths with evidence.\n"
            "4. 'What are your weaknesses?' — Choose a real weakness you're actively improving; "
            "show self-awareness and a development plan.\n"
            "5. 'Where do you see yourself in 5 years?' — Align with the company's growth; "
            "show ambition balanced with commitment to the role.\n"
            "6. 'Why are you leaving your current job?' — Stay positive; focus on growth, "
            "new challenges, alignment with your goals."
        ),
    },
    {
        "id": "hr_002",
        "category": "hr",
        "roles": ["all"],
        "title": "Salary Negotiation & Offer Discussion",
        "content": (
            "Salary questions require preparation and confidence. "
            "Research market ranges using Glassdoor, LinkedIn Salary, and industry surveys. "
            "When asked 'What are your salary expectations?': "
            "Give a researched range anchored slightly above your target. "
            "Example: 'Based on my research and experience level, I'm targeting ₹X–Y LPA, "
            "though I'm open to discussing the full compensation package.' "
            "Avoid anchoring too low. Highlight your value before discussing numbers. "
            "It's acceptable to ask about the budgeted range first."
        ),
    },
    {
        "id": "hr_003",
        "category": "hr",
        "roles": ["all"],
        "title": "Questions to Ask the Interviewer",
        "content": (
            "At the end of interviews, always prepare 2–3 thoughtful questions. "
            "Strong questions to ask:\n"
            "- 'What does success look like in this role after 90 days?'\n"
            "- 'What are the biggest challenges the team is currently facing?'\n"
            "- 'How would you describe the team culture and collaboration style?'\n"
            "- 'What opportunities are there for professional development?'\n"
            "- 'What are the next steps in the interview process?'\n"
            "Avoid asking about salary too early, or questions easily answered by the company website. "
            "Asking thoughtful questions signals genuine interest and preparation."
        ),
    },

    # ── EVALUATION CRITERIA ──────────────────────────────────────────────────
    {
        "id": "eval_001",
        "category": "evaluation",
        "roles": ["all"],
        "title": "Interview Answer Evaluation Criteria",
        "content": (
            "Strong interview answers are scored on these dimensions:\n"
            "1. Relevance (1-10): Does the answer directly address the question?\n"
            "2. Depth (1-10): Is the answer sufficiently detailed without being excessive?\n"
            "3. Structure (1-10): Is it organised (e.g., STAR for behavioural, clear logic for technical)?\n"
            "4. Technical Accuracy (1-10): For technical questions, is the information correct?\n"
            "5. Communication Clarity (1-10): Is the answer easy to follow?\n"
            "6. Example Quality (1-10): Are examples specific, quantified, and credible?\n"
            "Aggregate score = weighted average. Scores below 5 indicate significant gaps. "
            "7–8 indicates a strong candidate. 9–10 is exceptional. "
            "Feedback should be constructive: acknowledge what was good, then provide "
            "specific, actionable improvements."
        ),
    },
    {
        "id": "eval_002",
        "category": "evaluation",
        "roles": ["all"],
        "title": "Common Interview Mistakes to Avoid",
        "content": (
            "Top interview mistakes and how to avoid them:\n"
            "1. Vague answers — 'We worked on it as a team' without personal contribution. "
            "Fix: Always clarify YOUR specific role and actions.\n"
            "2. Negativity about past employers — raises red flags. Fix: Stay professional.\n"
            "3. Not quantifying results — 'I improved performance' vs 'I reduced load time by 40%'. "
            "Fix: Always attach metrics.\n"
            "4. Rambling — unfocused answers waste time. Fix: Use STAR or a clear 3-point structure.\n"
            "5. Not preparing questions — signals low interest. Fix: Prepare 3 thoughtful questions.\n"
            "6. Overconfidence or underconfidence — both are red flags. "
            "Fix: Be honest, specific, and composed.\n"
            "7. Failing to connect skills to job requirements. "
            "Fix: Study the JD and map your experiences to each requirement."
        ),
    },

    # ── EXPERIENCE-LEVEL GUIDANCE ────────────────────────────────────────────
    {
        "id": "exp_001",
        "category": "experience",
        "roles": ["all"],
        "title": "Fresher / Entry-Level Interview Expectations",
        "content": (
            "Freshers are evaluated on potential, learning ability, and fundamentals — not experience. "
            "Interviewers expect: solid understanding of core concepts, eagerness to learn, "
            "academic projects or internships as examples, and cultural fit. "
            "Technical freshers should know: OOP concepts, basic data structures, one primary language deeply. "
            "Behavioural examples can come from academic projects, hackathons, volunteer work, or part-time jobs. "
            "Strong fresher answer: 'In my final year project, I ...' with clear problem, approach, and outcome. "
            "Tip: Show intellectual curiosity — ask good questions, admit what you don't know "
            "but explain how you'd approach learning it."
        ),
    },
    {
        "id": "exp_002",
        "category": "experience",
        "roles": ["all"],
        "title": "Mid-Level (2-5 years) Interview Expectations",
        "content": (
            "Mid-level candidates are expected to work independently and deliver end-to-end. "
            "Interviewers look for: ownership of features/projects, technical depth beyond basics, "
            "cross-functional collaboration, mentoring junior members, and evidence of impact. "
            "Technical bar rises: system design basics, debugging complex issues, "
            "performance optimisation, code review experience. "
            "Behavioural bar rises: dealing with ambiguity, managing competing priorities, "
            "pushing back constructively when needed. "
            "Strong answer signals: past projects with quantifiable impact, "
            "proactive problem identification, and growth mindset."
        ),
    },
    {
        "id": "exp_003",
        "category": "experience",
        "roles": ["all"],
        "title": "Senior-Level (5+ years) Interview Expectations",
        "content": (
            "Senior candidates are evaluated on leadership, strategic thinking, and deep technical expertise. "
            "Interviewers expect: driving architectural decisions, influencing without authority, "
            "mentoring teams, aligning technical work with business goals, and managing trade-offs. "
            "Technical bar: complex system design, cross-team API contracts, scalability, security, "
            "technical debt management, and build-vs-buy decisions. "
            "Behavioural bar: driving org-level change, navigating conflict at the team/stakeholder level, "
            "growing other engineers. "
            "Strong answer signals: 'I designed the architecture for X that now serves Y million users', "
            "'I led the migration from A to B, reducing costs by Z%'."
        ),
    },
]


# ---------------------------------------------------------------------------
# Lightweight TF-IDF-style Retriever
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric chars."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_document(query_tokens: List[str], doc: dict) -> float:
    """Score a document by token overlap with the query, with role boost."""
    doc_text = (doc["title"] + " " + doc["content"] + " " + " ".join(doc["roles"])).lower()
    doc_tokens = set(_tokenize(doc_text))
    overlap = sum(1 for t in query_tokens if t in doc_tokens)
    # Slight boost for exact role match
    role_boost = sum(2 for role in doc["roles"] if role in " ".join(query_tokens))
    return overlap + role_boost


def retrieve(query: str, top_k: int = 4, category_filter: str = None) -> List[dict]:
    """
    Retrieve the top-k most relevant knowledge documents for a query.

    Parameters
    ----------
    query : str
        Free-text query (e.g., role + skills + question type).
    top_k : int
        Number of documents to return.
    category_filter : str, optional
        If set, restrict results to a specific category
        ('technical', 'behavioral', 'hr', 'evaluation', 'experience').

    Returns
    -------
    List[dict]
        Ranked list of matching knowledge documents.
    """
    query_tokens = _tokenize(query)
    candidates = KNOWLEDGE_DOCUMENTS
    if category_filter:
        candidates = [d for d in candidates if d["category"] == category_filter]

    scored = [(doc, _score_document(query_tokens, doc)) for doc in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored[:top_k] if score > 0]


def retrieve_for_profile(
    role: str,
    experience_level: str,
    skills: List[str],
    interview_type: str,
    top_k: int = 5,
) -> Tuple[str, List[dict]]:
    """
    Retrieve knowledge grounded to a candidate profile.

    Returns
    -------
    Tuple[str, List[dict]]
        (formatted_context_string, raw_documents)
    """
    query_parts = [role, experience_level] + skills + [interview_type]
    query = " ".join(query_parts)

    # Always include experience-level guidance
    exp_category = "experience"
    exp_docs = retrieve(experience_level + " " + role, top_k=1, category_filter=exp_category)

    # Retrieve category-specific docs
    cat_map = {
        "technical": "technical",
        "behavioral": "behavioral",
        "behavioural": "behavioral",
        "hr": "hr",
        "mixed": None,
    }
    cat = cat_map.get(interview_type.lower(), None)
    role_docs = retrieve(query, top_k=top_k - 1, category_filter=cat)

    # Evaluation guidance always included
    eval_docs = retrieve("evaluation criteria scoring", top_k=1, category_filter="evaluation")

    all_docs = {d["id"]: d for d in (exp_docs + role_docs + eval_docs)}.values()
    docs_list = list(all_docs)[:top_k + 1]

    context_parts = []
    for doc in docs_list:
        context_parts.append(f"[{doc['title']}]\n{doc['content']}")

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, docs_list
