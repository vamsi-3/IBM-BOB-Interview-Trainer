"""
Interview Trainer Agent — Main Entry Point
==========================================

Internship Problem Statement #22
Project: Interview Trainer Agent
Model  : meta-llama/llama-3-3-70b-instruct (IBM WatsonX, jp-tok)

Usage
-----
    python main.py                     # interactive mode (recommended)
    python main.py --api-key <KEY>     # pass API key via CLI flag

Environment variable (preferred):
    IBM_CLOUD_API_KEY=<your-key> python main.py
"""

import sys
import argparse
import os

from watsonx_client import WatsonXClient
from candidate_profile import collect_profile
from interview_engine import InterviewEngine


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║           AI INTERVIEW TRAINER AGENT                        ║
║           Powered by IBM WatsonX — Llama 3.3 70B            ║
║           RAG-based Personalised Interview Preparation       ║
╚══════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Interview Trainer Agent — IBM WatsonX"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="IBM Cloud API key (overrides IBM_CLOUD_API_KEY env variable)",
    )
    args = parser.parse_args()

    print(BANNER)

    # ── Resolve API key ───────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("IBM_CLOUD_API_KEY", "")
    if not api_key:
        print("  ERROR: IBM Cloud API key not found.")
        print("  Set the IBM_CLOUD_API_KEY environment variable or use --api-key.\n")
        print("  Example:")
        print("    Windows: set IBM_CLOUD_API_KEY=<your-key>  && python main.py")
        print("    Linux/Mac: IBM_CLOUD_API_KEY=<your-key> python main.py\n")
        sys.exit(1)

    # ── Initialise WatsonX client ─────────────────────────────────────────────
    print("  Connecting to IBM WatsonX (jp-tok)...")
    try:
        llm_client = WatsonXClient(api_key=api_key)
        # Quick token validation
        llm_client._ensure_token()
        print("  ✓ Connected to WatsonX successfully.\n")
    except Exception as exc:
        print(f"  ERROR: Could not connect to WatsonX: {exc}\n")
        print("  Please verify your IBM Cloud API key and network connectivity.")
        sys.exit(1)

    # ── Collect candidate profile ─────────────────────────────────────────────
    try:
        profile = collect_profile()
    except KeyboardInterrupt:
        print("\n\n  Interview setup cancelled. Goodbye!")
        sys.exit(0)

    # ── Run interview ─────────────────────────────────────────────────────────
    try:
        engine = InterviewEngine(llm_client=llm_client, profile=profile)
        engine.run()
    except KeyboardInterrupt:
        print("\n\n  Interview interrupted. Goodbye!")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  UNEXPECTED ERROR: {exc}")
        print("  Please check your API key, network connection, and try again.")
        sys.exit(1)

    print("\n  Thank you for using the AI Interview Trainer Agent. Good luck! 🎯\n")


if __name__ == "__main__":
    main()
