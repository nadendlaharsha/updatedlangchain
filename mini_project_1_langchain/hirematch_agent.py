"""
Mini Project 1 (LangChain): HireMatch - Automated Talent Screening & Interview Coordinator
Covers 5 Core LangChain Modules:
  1. Agent Setup & Provider Environment Initialization
  2. Multi-Provider Chat Models & Custom Tools (@tool)
  3. Message Abstractions (System, Human, AI, Tool) & Dialogue Management
  4. Structured Output with Pydantic (.with_structured_output)
  5. Agent Middleware (SummarizationMiddleware & HumanInTheLoopMiddleware)
"""

import os
import sys
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
# MODULE 1: Environment Setup
# =====================================================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")


# =====================================================================
# MODULE 2: Custom Tools (@tool Decorator)
# =====================================================================
@tool
def search_candidate_db(candidate_name: str) -> str:
    """Search candidate database for resume details and profile summary."""
    return (
        f"Candidate Record: {candidate_name}\n"
        f"- Experience: 5 years in Python, FastAPI, LangChain, and Agentic AI Architecture.\n"
        f"- Past Role: Senior AI Developer at CloudTech.\n"
        f"- Education: B.S. Computer Science.\n"
        f"- Expected Salary: $120,000/year.\n"
        f"- Email: {candidate_name.lower().replace(' ', '')}@example.com"
    )

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
    candidate_name: str = Field(description="Full name of the candidate")
    candidate_email: str = Field(description="Email address of the candidate")
    skill_match_score: int = Field(description="Technical skill match score out of 100")
    experience_level: Literal["Junior", "Mid-Level", "Senior", "Lead"] = Field(description="Assessed experience level")
    expected_salary: str = Field(description="Expected salary mentioned or assessed")
    recommendation: Literal["Hire", "Interview", "Reject", "Hold"] = Field(description="Recruitment decision recommendation")
    key_strengths: List[str] = Field(description="List of key candidate strengths")
    summary: str = Field(description="Executive evaluation summary")


# =====================================================================
# MODULE 1 & 5: Agent Initialization with Middleware
# =====================================================================
print("--------------------------------------------------")
print("🚀 Initializing HireMatch Agent with Middleware...")
print("--------------------------------------------------")

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[search_candidate_db, check_interviewer_availability, send_interview_invite, send_rejection_notice],
    checkpointer=InMemorySaver(),
    middleware=[
        # Summarize conversation context when token threshold is reached
        SummarizationMiddleware(
            model="groq:openai/gpt-oss-120b",
            trigger=("tokens", 600),
            keep=("tokens", 250)
        ),
        # Human-in-the-Loop: Interrupt on critical actions (invites & rejections)
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_interview_invite": {"allowed_decisions": ["approve", "edit", "reject"]},
                "send_rejection_notice": {"allowed_decisions": ["approve", "edit", "reject"]},
                "search_candidate_db": False,
                "check_interviewer_availability": False,
            }
        )
    ]
)


# =====================================================================
# MODULE 3 & 5: Running Agent & Handling Human-in-the-Loop Interrupts
# =====================================================================
def run_screening_demo():
    config = {"configurable": {"thread_id": "screening_candidate_alex"}}
    
    system_prompt = (
        "You are HireMatch AI, a professional recruitment coordinator. "
        "Your task is to search candidate records, verify interviewer availability, "
        "and invite top candidates for interviews."
    )
    
    user_request = (
        "Please look up candidate 'Alex Rivera'. If qualified, check interviewer 'Sarah Connor' "
        "availability for '2026-09-01' and send an interview invitation for 10:00 AM EST."
    )

    print("\n📩 [USER REQUEST]:", user_request)
    print("\n⚙️ [STEP 1]: Running HireMatch Agent...")

    # First invocation
    result = agent.invoke(
        {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_request)]},
        config=config
    )

    # Check for Human-In-The-Loop Interrupt
    if "__interrupt__" in result:
        print("\n⚠️ [HUMAN-IN-THE-LOOP INTERRUPT DETECTED]")
        interrupt_info = result["__interrupt__"][0].value
        action_req = interrupt_info["action_requests"][0]
        
        print(f"  Pending Action : {action_req['name']}")
        print(f"  Arguments      : {action_req['args']}")
        print(f"  Allowed Options: {interrupt_info['review_configs'][0]['allowed_decisions']}")
        
        # Simulating Human Approval
        print("\n👤 [HUMAN RECRUITER ACTION]: Approving the pending interview invitation...")
        
        final_result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )
        
        print("\n✅ [FINAL AGENT RESPONSE]:")
        print(final_result["messages"][-1].content)
        return final_result
    else:
        print("\n✅ [AGENT RESPONSE]:", result["messages"][-1].content)
        return result


# =====================================================================
# MODULE 4: Generating Structured Candidate Evaluation Report
# =====================================================================
def generate_structured_report():
    print("\n--------------------------------------------------")
    print("📊 [REPORT] Generating Pydantic Candidate Evaluation Card...")
    print("--------------------------------------------------")

    model = init_chat_model("groq:openai/gpt-oss-120b")
    structured_llm = model.with_structured_output(CandidateEvaluation)

    candidate_text = (
        "Alex Rivera has 5 years of experience in Python, FastAPI, LangChain, and AI Agents. "
        "Expected salary is $120,000/year. Email: alexrivera@example.com. "
        "Strong technical skills and past leadership at CloudTech. Highly recommended for Senior AI role."
    )

    evaluation = structured_llm.invoke(
        f"Evaluate the candidate based on these details: {candidate_text}"
    )

    print("\n✨ [STRUCTURED OUTPUT]:")
    print(f"Name              : {evaluation.candidate_name}")
    print(f"Email             : {evaluation.candidate_email}")
    print(f"Skill Match Score : {evaluation.skill_match_score}/100")
    print(f"Experience Level  : {evaluation.experience_level}")
    print(f"Expected Salary   : {evaluation.expected_salary}")
    print(f"Recommendation    : {evaluation.recommendation}")
    print(f"Key Strengths     : {', '.join(evaluation.key_strengths)}")
    print(f"Summary           : {evaluation.summary}")


if __name__ == "__main__":
    run_screening_demo()
    generate_structured_report()
