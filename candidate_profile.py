"""
Candidate Profile Module
-------------------------
Collects structured information about the candidate via a CLI wizard.
"""

from dataclasses import dataclass, field
from typing import List, Optional


EXPERIENCE_LEVELS = ["fresher", "junior", "mid-level", "senior", "lead/principal"]
INTERVIEW_TYPES = ["technical", "behavioral", "hr", "mixed"]


@dataclass
class CandidateProfile:
    name: str
    experience_level: str          # e.g., "fresher", "mid-level", "senior"
    target_role: str               # e.g., "Software Engineer", "Data Scientist"
    skills: List[str]              # e.g., ["Python", "Django", "SQL"]
    resume_text: str               # Paste or summarise resume
    job_description: str           # JD text, or empty string if not provided
    interview_type: str            # "technical" | "behavioral" | "hr" | "mixed"
    years_of_experience: Optional[int] = None

    def summary(self) -> str:
        """Return a compact text summary of the profile for prompt injection."""
        skills_str = ", ".join(self.skills) if self.skills else "Not specified"
        jd_note = (
            "Provided (see below)"
            if self.job_description.strip()
            else "Not provided"
        )
        return (
            f"Candidate Name       : {self.name}\n"
            f"Experience Level     : {self.experience_level}"
            + (f" ({self.years_of_experience} years)" if self.years_of_experience else "")
            + f"\nTarget Role          : {self.target_role}\n"
            f"Key Skills           : {skills_str}\n"
            f"Interview Type       : {self.interview_type}\n"
            f"Job Description      : {jd_note}"
        )


def _prompt(label: str, default: str = "") -> str:
    """Ask for input with an optional default."""
    default_hint = f" [{default}]" if default else ""
    value = input(f"  {label}{default_hint}: ").strip()
    return value if value else default


def _prompt_choice(label: str, choices: List[str]) -> str:
    """Display a numbered menu and return the chosen value."""
    print(f"\n  {label}")
    for i, choice in enumerate(choices, 1):
        print(f"    {i}. {choice}")
    while True:
        raw = input("  Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(choices)}.")


def _prompt_multiline(label: str) -> str:
    """Collect a multi-line text block (blank line to finish)."""
    print(f"\n  {label}")
    print("  (Paste text then press Enter twice to finish)")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def collect_profile() -> CandidateProfile:
    """
    Interactive CLI wizard to collect candidate profile information.

    Returns
    -------
    CandidateProfile
        Fully populated profile object.
    """
    print("\n" + "═" * 60)
    print("  INTERVIEW TRAINER AGENT — Candidate Profile Setup")
    print("═" * 60)

    name = _prompt("Your full name")
    experience_level = _prompt_choice("Select your experience level:", EXPERIENCE_LEVELS)

    years_str = _prompt("Years of experience (press Enter to skip)", "")
    years = int(years_str) if years_str.isdigit() else None

    target_role = _prompt("Target job role (e.g., Software Engineer, Data Scientist)")
    skills_raw = _prompt(
        "Your key skills, comma-separated (e.g., Python, SQL, Machine Learning)"
    )
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

    interview_type = _prompt_choice("Interview type:", INTERVIEW_TYPES)

    print("\n  RESUME — paste a summary of your resume or key highlights.")
    resume_text = _prompt_multiline("Resume text")

    print(
        "\n  JOB DESCRIPTION — paste the JD you are preparing for (optional)."
    )
    use_jd = input("  Do you have a job description to paste? (y/n): ").strip().lower()
    if use_jd == "y":
        job_description = _prompt_multiline("Job description")
    else:
        job_description = ""

    profile = CandidateProfile(
        name=name,
        experience_level=experience_level,
        target_role=target_role,
        skills=skills,
        resume_text=resume_text,
        job_description=job_description,
        interview_type=interview_type,
        years_of_experience=years,
    )

    print("\n" + "─" * 60)
    print("  Profile collected successfully. Here is a summary:\n")
    print(profile.summary())
    print("─" * 60)

    confirm = input("\n  Looks good? Press Enter to start the interview, or type 'edit' to redo: ").strip()
    if confirm.lower() == "edit":
        return collect_profile()

    return profile
