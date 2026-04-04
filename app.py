from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from env.environment import PaymentRecoveryEnv
from env.models import Action
from env.tasks import TASKS
from grader.grader import Grader

app = FastAPI(title="Payment Recovery Env")
environments = {}
grader = Grader()

VALID_TASKS = list(TASKS.keys())  # ["easy", "medium", "hard"]


class StepRequest(BaseModel):
    task_name: str
    message: str


class GraderRequest(BaseModel):
    task_name: str
    conversation_history: list[str]
    amount_recovered: float
    invoice_amount: float
    steps_taken: int


def _get_obs_dict(obs):
    return {
        "client_name": obs.client_name,
        "invoice_amount": obs.invoice_amount,
        "days_overdue": obs.days_overdue,
        "client_status": obs.client_status,
        "client_personality": obs.client_personality,
        "background": obs.background,
        "conversation_history": obs.conversation_history,
        "current_step": obs.current_step,
        "amount_recovered": obs.amount_recovered,
        "is_done": obs.is_done,
    }


@app.get("/")
def read_root():
    return JSONResponse({
        "name": "payment-recovery-env",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grader", "/baseline"]
    })


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "environment": "payment-recovery-env"})


@app.get("/tasks")
def get_tasks():
    task_list = []
    for t_id, t_data in TASKS.items():
        descriptions = {
            "easy": "Recover $1000 from cooperative client, 7 days overdue",
            "medium": "Recover $3000 from ghosting client, 30 days overdue",
            "hard": "Negotiate disputed $8000 invoice, 90 days overdue",
        }
        task_list.append({
            "id": t_id,
            "description": descriptions[t_id],
            "difficulty": t_id,
            "invoice_amount": t_data["invoice_amount"],
            "client_status": t_data["client_status"],
            "max_steps": 5,
            "action_space": "Natural language message (str)",
            "observation_space": "Client profile, invoice details, conversation history, current step, amount recovered, completion status"
        })
    return JSONResponse(task_list)


@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
        task_name = body.get("task_name", "easy") if isinstance(body, dict) else "easy"
    except Exception:
        task_name = "easy"

    if task_name not in VALID_TASKS:
        task_name = "easy"

    env = PaymentRecoveryEnv(task_name)
    obs = env.reset()
    environments[task_name] = env

    return JSONResponse(_get_obs_dict(obs))


@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
        task_name = body.get("task_name", "easy")
        message = body.get("message", "")
    except Exception:
        return JSONResponse({"error": "Invalid request body", "done": True}, status_code=200)

    if not message:
        return JSONResponse({"error": "message field is required", "done": True}, status_code=200)

    if task_name not in VALID_TASKS:
        task_name = "easy"

    if task_name not in environments:
        env = PaymentRecoveryEnv(task_name)
        env.reset()
        environments[task_name] = env

    env = environments[task_name]
    action = Action(message=message)
    obs, reward, done, info = env.step(action)

    return JSONResponse({
        "observation": _get_obs_dict(obs),
        "reward": {"score": reward.score, "reason": reward.reason},
        "done": done,
        "info": info
    })


@app.get("/state")
def get_state(task_name: str = "easy"):
    if task_name not in environments:
        env = PaymentRecoveryEnv(task_name if task_name in VALID_TASKS else "easy")
        obs = env.reset()
        environments[task_name] = env
        return JSONResponse(_get_obs_dict(obs))
    return JSONResponse(_get_obs_dict(environments[task_name].state()))


@app.post("/grader")
async def run_grader(request: Request):
    try:
        body = await request.json()
        task_name = body.get("task_name", "easy")
        conversation_history = body.get("conversation_history", [])
        amount_recovered = float(body.get("amount_recovered", 0.0))
        invoice_amount = float(body.get("invoice_amount", 1000.0))
        steps_taken = int(body.get("steps_taken", 0))
    except Exception:
        return JSONResponse({"score": 0.0, "explanation": "Invalid request body"}, status_code=200)

    score = grader.grade(task_name, conversation_history, amount_recovered, invoice_amount, steps_taken)

    ratio = (amount_recovered / invoice_amount) if invoice_amount > 0 else 0
    po = 0.5 if ratio >= 1.0 else (0.3 if ratio >= 0.5 else (0.1 if ratio > 0.0 else 0.0))

    neg_score, tone_score = 0.0, 0.0
    for m in conversation_history:
        if m.startswith("Agent:"):
            ml = m.lower()
            neg_score += sum(0.05 for kw in ["value", "relationship", "mutual", "resolve", "commitment", "appreciate", "understand"] if kw in ml)
            tone_score += sum(0.05 for kw in ["appreciate", "understand", "kindly", "request", "please"] if kw in ml)
            tone_score -= sum(0.1 for kw in ["sue", "legal", "useless", "fraud"] if kw in ml)

    neg_score = min(neg_score, 0.3)
    tone_score = min(tone_score, 0.2)
    explanation = grader.explain_score(score, po, tone_score, neg_score)

    return JSONResponse({"score": score, "explanation": explanation})


@app.post("/baseline")
def run_baseline():
    mock_messages = {
        "easy": [
            "We appreciate your business. I am following up on invoice #123. Could you kindly confirm the payment status?",
            "We understand things come up. We value our relationship and would appreciate if you could arrange the payment soon.",
            "Please confirm once done. We understand and thank you for your flexibility.",
        ],
        "medium": [
            "We have noticed your payment is 30 days overdue. We value our relationship and want to resolve this together.",
            "We understand things take time. Can we arrange a timeline or installment plan?",
            "Perhaps we can settle this with a partial payment arrangement?",
        ],
        "hard": [
            "Hello, we are following up on invoice #456. We need to resolve this outstanding balance.",
            "We understand there are concerns about the services. We want to be flexible and resolve this mutually.",
            "We are open to a compromise. Would a partial settlement help us reach an arrangement?",
            "Let's finalize at 60% of the original invoice. Please confirm so we can close this.",
        ]
    }

    results = {}
    for task_name in VALID_TASKS:
        env = PaymentRecoveryEnv(task_name)
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < len(mock_messages[task_name]):
            obs, reward, done, _ = env.step(Action(message=mock_messages[task_name][steps]))
            steps += 1

        score = grader.grade(task_name, obs.conversation_history, obs.amount_recovered, obs.invoice_amount, steps)
        results[task_name] = {
            "score": score,
            "steps": steps,
            "amount_recovered": obs.amount_recovered,
            "invoice_amount": obs.invoice_amount,
        }

    return JSONResponse(results)
