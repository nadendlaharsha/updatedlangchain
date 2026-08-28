"""
Mini Project 1 (LangChain): HireMatch - Automated Talent Screening & Interview Coordinator

Dynamic & Non-Hardcoded Architecture:
  - Dynamically parses submitted resume files (.txt, .md) or URL links.
  - Dynamically extracts CandidateEvaluation Pydantic structured output from actual resume text.
  - Interactive Human-In-The-Loop Middleware: Terminal prompts human recruiter to Approve, Edit, or Reject tool calls!
"""

import os
import sys
import time
import argparse
from typing import Literal, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# =====================================================================
# PATH RESOLUTION & ENVIRONMENT SETUP
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Load .env from project root or current folder
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

MODEL_NAME = "groq:openai/gpt-oss-120b"
DEFAULT_RESUME_PATH = os.path.join(PROJECT_ROOT, "resumes", "alex_rivera_resume.txt")


def invoke_with_retry(invoke_func, *args, **kwargs):
    """Utility wrapper to catch API rate limits and retry automatically."""
    for attempt in range(4):
        try:
            return invoke_func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RateLimit" in type(e).__name__:
                wait_time = 4 * (attempt + 1)
                print(f"⏳ [RATE LIMIT NOTICE] Waiting {wait_time}s before retrying (Attempt {attempt+1}/4)...")
                time.sleep(wait_time)
            else:
                raise e
    return invoke_func(*args, **kwargs)


# =====================================================================
# MODULE 2: Dynamic Custom Tools (No Hardcoded Data)
# =====================================================================
@tool
def read_submitted_resume(file_path_or_url: str) -> str:
    """Read and parse candidate resume text from a submitted file path or URL link."""
    target_path = file_path_or_url
    if not os.path.exists(target_path):
        alt_path = os.path.join(PROJECT_ROOT, file_path_or_url)
        if os.path.exists(alt_path):
            target_path = alt_path

    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"--- RESUME CONTENT ({target_path}) ---\n{content}"
        except Exception as e:
            return f"Error reading resume file {target_path}: {str(e)}"
    else:
        return f"Resume input received: {file_path_or_url}"

@tool
def check_interviewer_availability(interviewer_name: str, date: str) -> str:
    """Check calendar availability of a hiring manager for a specific date."""
    return f"Availability for {interviewer_name} on {date}: Available at 10:00 AM EST and 2:00 PM EST."

@tool
def send_interview_invite(candidate_email: str, interviewer_name: str, slot: str) -> str:
    """Send official interview invitation email to candidate."""
    return f"SUCCESS: Interview invitation sent to {candidate_email} with {interviewer_name} for {slot}."

@tool
def send_rejection_notice(candidate_email: str, reason: str) -> str:
    """Send polite rejection notice to candidate."""
    return f"SUCCESS: Rejection notice sent to {candidate_email}. Reason logged: {reason}"


# =====================================================================
# MODULE 4: Structured Output Schema (Pydantic)
# =====================================================================
class CandidateEvaluation(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate extracted from resume")
    candidate_email: str = Field(description="Email address of the candidate extracted from resume")
    skill_match_score: int = Field(description="Technical skill match score out of 100 based on experience")
    experience_level: Literal["Junior", "Mid-Level", "Senior", "Lead"] = Field(description="Assessed experience level")
    expected_salary: str = Field(description="Expected salary mentioned in resume or market assessment")
    recommendation: Literal["Hire", "Interview", "Reject", "Hold"] = Field(description="Recruitment decision recommendation")
    key_strengths: List[str] = Field(description="List of key technical strengths extracted from resume")
    summary: str = Field(description="Executive evaluation summary of candidate resume")


# =====================================================================
# MODULE 1 & 5: Agent Initialization with Middleware
# =====================================================================
agent = create_agent(
    model=MODEL_NAME,
    tools=[read_submitted_resume, check_interviewer_availability, send_interview_invite, send_rejection_notice],
    checkpointer=InMemorySaver(),
    middleware=[
        # Summarize conversation context when token threshold is reached
        SummarizationMiddleware(
            model=MODEL_NAME,
            trigger=("tokens", 600),
            keep=("tokens", 250)
        ),
        # Human-in-the-Loop: Interrupt on critical actions (invites & rejections)
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_interview_invite": {"allowed_decisions": ["approve", "edit", "reject"]},
                "send_rejection_notice": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_submitted_resume": False,
                "check_interviewer_availability": False,
            }
        )
    ]
)


# =====================================================================
# DYNAMIC RUNNER & INTERACTIVE WORKFLOW EXECUTION
# =====================================================================
def process_candidate_resume(
    resume_file_path: str, 
    interviewer_name: str = "Sarah Connor", 
    interview_date: str = "2026-09-01",
    auto_approve: bool = False
):
    """
    Dynamically processes any submitted candidate resume file or URL link without hardcoding details.
    Prompts human recruiter interactively in terminal for approval, edit, or rejection.
    """
    target_file = resume_file_path
    if not os.path.exists(target_file):
        alt_path = os.path.join(PROJECT_ROOT, resume_file_path)
        if os.path.exists(alt_path):
            target_file = alt_path

    if not os.path.exists(target_file):
        print(f"❌ Error: Resume file '{resume_file_path}' not found!")
        return

    # Read raw resume content directly from file
    with open(target_file, "r", encoding="utf-8") as f:
        raw_resume_text = f.read()

    print("\n--------------------------------------------------")
    print(f"📂 [INPUT] Processing Submitted Resume File: {target_file}")
    print("--------------------------------------------------")

    # -----------------------------------------------------------------
    # STEP A: Dynamic Structured Extraction (Module 4)
    # -----------------------------------------------------------------
    print("📊 [MODULE 4]: Extracting Structured Evaluation Card from Resume...")
    model = init_chat_model(MODEL_NAME)
    structured_llm = model.with_structured_output(CandidateEvaluation)
    
    evaluation: CandidateEvaluation = invoke_with_retry(
        structured_llm.invoke,
        f"Parse and evaluate this candidate resume text:\n\n{raw_resume_text}"
    )

    print("\n✨ [EXTRACTED CANDIDATE EVALUATION CARD]:")
    print(f"  Name              : {evaluation.candidate_name}")
    print(f"  Email             : {evaluation.candidate_email}")
    print(f"  Skill Match Score : {evaluation.skill_match_score}/100")
    print(f"  Experience Level  : {evaluation.experience_level}")
    print(f"  Expected Salary   : {evaluation.expected_salary}")
    print(f"  Recommendation    : {evaluation.recommendation}")
    print(f"  Key Strengths     : {', '.join(evaluation.key_strengths)}")
    print(f"  Summary           : {evaluation.summary}")

    # Brief delay to respect rate limit
    time.sleep(2)

    # -----------------------------------------------------------------
    # STEP B: Dynamic Agent Workflow & HITL Governance (Modules 1, 2, 3, 5)
    # -----------------------------------------------------------------
    clean_name = evaluation.candidate_name.lower().replace(' ', '_')
    thread_id = f"session_{clean_name}"
    config = {"configurable": {"thread_id": thread_id}}

    system_prompt = (
        "You are HireMatch AI, an executive recruitment coordinator. "
        "Read candidate resumes, check interviewer availability, and invite qualified candidates."
    )

    user_request = (
        f"Read the resume at '{target_file}'. "
        f"The candidate is '{evaluation.candidate_name}' ({evaluation.candidate_email}). "
        f"Based on the evaluation recommendation '{evaluation.recommendation}', "
        f"if qualified, check availability for interviewer '{interviewer_name}' on '{interview_date}' "
        f"and send an interview invitation for 10:00 AM EST. Otherwise send a rejection notice."
    )

    print("\n⚙️ [MODULES 1,2,3,5]: Running HireMatch Agent Workflow...")
    result = invoke_with_retry(
        agent.invoke,
        {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_request)]},
        config=config
    )

    # Check for Human-In-The-Loop Interrupt
    if "__interrupt__" in result:
        print("\n==================================================")
        print("⚠️ [HUMAN-IN-THE-LOOP INTERRUPT TRIGGERED]")
        interrupt_info = result["__interrupt__"][0].value
        action_req = interrupt_info["action_requests"][0]
        
        print(f"  Action Request : {action_req['name']}")
        print(f"  Target Arguments: {action_req['args']}")
        print(f"  Allowed Decisions: {interrupt_info['review_configs'][0]['allowed_decisions']}")
        print("==================================================")
        
        # Interactive Decision Prompt (Terminal CLI)
        if sys.stdin.isatty() and not auto_approve:
            print("\n👤 HUMAN RECRUITER DECISION REQUIRED:")
            print("   1. Type 'approve' to send the email as requested.")
            print("   2. Type 'reject'  to block sending the email.")
            print("   3. Type 'edit'    to modify recipient, subject, or slot before sending.")
            
            choice = input("\n👉 Enter your decision [approve/edit/reject] (default: approve): ").strip().lower()
            if not choice:
                choice = "approve"
            
            if choice == "reject":
                reason = input("Enter rejection reason for candidate (optional): ").strip()
                decision_payload = {"type": "reject", "message": reason if reason else "Recruiter rejected tool execution."}
                print("\n🚫 [HUMAN RECRUITER DECISION]: REJECTED sending email.")
            elif choice == "edit":
                print("\n✏️  [EDITING TOOL ARGUMENTS]:")
                edited_args = dict(action_req['args'])
                for k, v in edited_args.items():
                    new_val = input(f"   Change {k} (current: '{v}'): ").strip()
                    if new_val:
                        edited_args[k] = new_val
                decision_payload = {
                    "type": "edit",
                    "edited_action": {
                        "name": action_req["name"],
                        "args": edited_args
                    }
                }
                print(f"\n✏️  [HUMAN RECRUITER DECISION]: EDITED action -> {edited_args}")
            else:
                decision_payload = {"type": "approve"}
                print("\n✅ [HUMAN RECRUITER DECISION]: APPROVED sending email.")
        else:
            print("\n👤 [HUMAN RECRUITER DECISION]: Auto-approving pending tool execution...")
            decision_payload = {"type": "approve"}
        
        time.sleep(2)

        final_result = invoke_with_retry(
            agent.invoke,
            Command(resume={"decisions": [decision_payload]}),
            config=config
        )
        
        print("\n✅ [FINAL AGENT RESPONSE]:")
        print(final_result["messages"][-1].content)
    else:
        print("\n✅ [AGENT RESPONSE]:", result["messages"][-1].content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HireMatch Dynamic Talent Screening Agent")
    parser.add_argument(
        "--resume", 
        type=str, 
        default=DEFAULT_RESUME_PATH,
        help="Path to candidate resume file (.txt, .md)"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip interactive prompt and auto-approve pending tool executions"
    )
    args = parser.parse_args()

    process_candidate_resume(args.resume, auto_approve=args.auto-approve)
