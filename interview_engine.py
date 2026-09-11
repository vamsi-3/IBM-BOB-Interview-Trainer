"""
Interview Engine
-----------------
Orchestrates the interview session: question generation, answer collection,
evaluation, and final report generation.
"""

import re
from typing import List, Optional

from watsonx_client import WatsonXClient
from candidate_profile import CandidateProfile
from knowledge_base.rag_knowledge import retrieve_for_profile
from prompt_templates import (
    build_question_prompt,
    build_evaluation_prompt,
    build_final_report_prompt,
)


# Default number of questions per interview type
QUESTION_COUNT = {
    "technical": 5,
    "behavioral": 5,
    "hr": 4,
    "mixed": 6,
}


def _extract_score(evaluation_text: str) -> Optional[float]:
    """Parse the numeric score from evaluation text like 'SCORE: 7/10'."""
    match = re.search(r"SCORE\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*10", evaluation_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Fallback: look for standalone number
    match = re.search(r"\b([1-9]|10)\s*/\s*10\b", evaluation_text)
    if match:
        return float(match.group(1))
    return None


def _detect_question_type(question_number: int, interview_type: str, total: int) -> str:
    """Infer the question type for a given question number in a mixed interview."""
    if interview_type != "mixed":
        return interview_type
    # Mixed: first ~40% technical, next ~40% behavioral, last ~20% HR
    tech_cutoff = max(1, round(total * 0.4))
    behav_cutoff = max(tech_cutoff + 1, round(total * 0.8))
    if question_number <= tech_cutoff:
        return "technical"
    elif question_number <= behav_cutoff:
        return "behavioral"
    else:
        return "hr"


def _print_divider(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _print_section(title: str) -> None:
    _print_divider("═")
    print(f"  {title}")
    _print_divider("═")


class InterviewEngine:
    """
    Runs a complete interview session for a candidate.

    Lifecycle
    ---------
    1. retrieve RAG knowledge for the candidate profile
    2. generate questions one-by-one
    3. collect and evaluate each answer
    4. produce a final performance report
    """

    def __init__(self, llm_client: WatsonXClient, profile: CandidateProfile):
        self.llm = llm_client
        self.profile = profile
        self.qa_history: List[dict] = []
        self.total_questions = QUESTION_COUNT.get(profile.interview_type.lower(), 5)

        # Retrieve RAG knowledge once up-front for this profile
        print("\n  🔍 Retrieving personalised knowledge base...")
        self.rag_context, self.rag_docs = retrieve_for_profile(
            role=profile.target_role,
            experience_level=profile.experience_level,
            skills=profile.skills,
            interview_type=profile.interview_type,
            top_k=6,
        )
        print(f"  ✓ Retrieved {len(self.rag_docs)} relevant knowledge documents.")

    # ── Question Generation ───────────────────────────────────────────────────

    def _generate_question(self, question_number: int) -> str:
        """Generate the next interview question via LLM."""
        previous_questions = [item["question"] for item in self.qa_history]
        q_type = _detect_question_type(
            question_number, self.profile.interview_type, self.total_questions
        )

        prompt = build_question_prompt(
            profile=self.profile,
            rag_context=self.rag_context,
            question_number=question_number,
            total_questions=self.total_questions,
            previous_questions=previous_questions,
            interview_type=q_type,
        )

        return self.llm.generate(prompt, max_new_tokens=300, temperature=0.7)

    # ── Answer Evaluation ─────────────────────────────────────────────────────

    def _evaluate_answer(self, question: str, answer: str, q_type: str) -> str:
        """Evaluate the candidate's answer via LLM."""
        prompt = build_evaluation_prompt(
            profile=self.profile,
            question=question,
            candidate_answer=answer,
            rag_context=self.rag_context,
            question_type=q_type,
        )
        return self.llm.generate(prompt, max_new_tokens=900, temperature=0.5)

    # ── Final Report ──────────────────────────────────────────────────────────

    def _generate_final_report(self) -> str:
        """Generate the comprehensive final interview report."""
        prompt = build_final_report_prompt(
            profile=self.profile,
            qa_history=self.qa_history,
            rag_context=self.rag_context,
        )
        return self.llm.generate(prompt, max_new_tokens=1200, temperature=0.4)

    # ── Session Runner ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the full interactive interview session."""

        _print_section(
            f"INTERVIEW STARTED  |  {self.profile.name}  |  {self.profile.target_role}  "
            f"|  {self.profile.interview_type.upper()}"
        )
        print(
            f"\n  This interview has {self.total_questions} questions.\n"
            f"  After each question, type your answer and press Enter twice to submit.\n"
            f"  Type 'skip' to skip a question (will count as unanswered).\n"
            f"  Type 'quit' to end early and receive a partial report.\n"
        )

        for q_num in range(1, self.total_questions + 1):
            q_type = _detect_question_type(q_num, self.profile.interview_type, self.total_questions)

            _print_divider()
            print(f"\n  Question {q_num} of {self.total_questions}  [{q_type.upper()}]")
            print("  Generating question...\n")

            try:
                question = self._generate_question(q_num)
            except Exception as exc:
                print(f"  [ERROR] Could not generate question: {exc}")
                continue

            print(f"  ❓ {question}\n")
            _print_divider("·")

            # Collect answer
            print("  Your answer (press Enter twice to submit, or type 'skip'/'quit'):")
            answer_lines = []
            while True:
                line = input()
                if line.lower() == "quit":
                    print("\n  Interview ended early. Generating partial report...\n")
                    if self.qa_history:
                        self._display_final_report()
                    return
                if line.lower() == "skip":
                    answer_lines = ["[Skipped]"]
                    break
                if line == "" and answer_lines and answer_lines[-1] == "":
                    break
                answer_lines.append(line)

            candidate_answer = "\n".join(answer_lines).strip()
            if candidate_answer == "[Skipped]" or not candidate_answer:
                print("  ⏭  Question skipped.\n")
                self.qa_history.append({
                    "question": question,
                    "answer": "[Skipped]",
                    "evaluation": "Question was skipped.",
                    "score": 0,
                    "type": q_type,
                })
                continue

            # Evaluate
            print("\n  ⚙️  Evaluating your answer...\n")
            try:
                evaluation = self._evaluate_answer(question, candidate_answer, q_type)
            except Exception as exc:
                print(f"  [ERROR] Evaluation failed: {exc}")
                evaluation = "Evaluation unavailable due to an error."

            score = _extract_score(evaluation)

            self.qa_history.append({
                "question": question,
                "answer": candidate_answer,
                "evaluation": evaluation,
                "score": score,
                "type": q_type,
            })

            # Display evaluation
            _print_divider()
            print(f"\n  📊 EVALUATION — Question {q_num}\n")
            print(evaluation)
            print()

            if q_num < self.total_questions:
                input("  Press Enter to continue to the next question... ")
            else:
                print("\n  ✅ All questions completed!\n")

        self._display_final_report()

    def _display_final_report(self) -> None:
        """Generate and print the final interview report."""
        _print_section("GENERATING FINAL INTERVIEW REPORT")
        print("  Please wait...\n")

        try:
            report = self._generate_final_report()
        except Exception as exc:
            print(f"  [ERROR] Could not generate final report: {exc}")
            self._print_fallback_report()
            return

        _print_divider("═")
        print(report)
        _print_divider("═")

        # Offer to save
        save = input("\n  Save this report to a file? (y/n): ").strip().lower()
        if save == "y":
            filename = f"interview_report_{self.profile.name.replace(' ', '_').lower()}.txt"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"CANDIDATE: {self.profile.name}\n")
                    f.write(f"ROLE: {self.profile.target_role}\n\n")
                    f.write(report)
                print(f"  ✓ Report saved to: {filename}")
            except IOError as e:
                print(f"  Could not save file: {e}")

    def _print_fallback_report(self) -> None:
        """Minimal fallback report if LLM call fails."""
        scores = [item["score"] for item in self.qa_history if isinstance(item.get("score"), (int, float))]
        avg = round(sum(scores) / len(scores), 1) if scores else "N/A"
        _print_divider("═")
        print(f"\n  INTERVIEW SUMMARY FOR {self.profile.name}")
        print(f"  Role: {self.profile.target_role}")
        print(f"  Questions answered: {len(self.qa_history)}")
        print(f"  Average score: {avg}/10\n")
        for i, item in enumerate(self.qa_history, 1):
            print(f"  Q{i}: {item['question'][:80]}...")
            print(f"      Score: {item.get('score', 'N/A')}/10")
        _print_divider("═")
