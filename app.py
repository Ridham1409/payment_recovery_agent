from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env.environment import PaymentRecoveryEnv
from env.models import Action
from env.tasks import TASKS
from grader.grader import Grader

app = FastAPI(title="Payment Recovery Env")
environments = {}
grader = Grader()

class ResetRequest(BaseModel):
    task_name: str

class StepRequest(BaseModel):
    task_name: str
    message: str

class GraderRequest(BaseModel):
    task_name: str
    conversation_history: list[str]
    amount_recovered: float
    invoice_amount: float
    steps_taken: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Payment Recovery & Negotiation Environment API. Check /docs for endpoints. System is Running."}

@app.get("/health")
def health():
    return {"status": "ok", "environment": "payment-recovery-env"}

@app.get("/tasks")
def get_tasks():
    tasks = []
    for t_id, t_data in TASKS.items():
        if t_id == "easy":
            desc = "Recover $1000 from cooperative client, 7 days overdue"
            diff = "easy"
        elif t_id == "medium":
            desc = "Recover $3000 from ghosting client, 30 days overdue"
            diff = "medium"
        else:
            desc = "Negotiate disputed $8000 invoice, 90 days overdue"
            diff = "hard"
            
        tasks.append({
            "id": t_id,
            "description": desc,
            "difficulty": diff,
            "invoice_amount": t_data["invoice_amount"],
            "client_status": t_data["client_status"],
            "action_space": "Natural language message (str)",
            "observation_space": "Client profile, invoice details, conversation history, current step, amount recovered, completion status"
        })
    return tasks

@app.post("/reset")
def reset(req: ResetRequest):
    if req.task_name not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{req.task_name}'. Valid tasks: easy, medium, hard")
    
    environments[req.task_name] = PaymentRecoveryEnv(req.task_name)
    obs = environments[req.task_name].reset()
    return {"status": "ok", "task_name": req.task_name, "observation": obs}

@app.post("/step")
def step(req: StepRequest):
    if req.task_name not in environments:
        raise HTTPException(status_code=404, detail="Environment not initialized. Call /reset first.")
    
    env = environments[req.task_name]
    action = Action(message=req.message)
    obs, reward, done, info = env.step(action)
    
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state(task_name: str):
    if task_name not in environments:
        raise HTTPException(status_code=404, detail="Environment not initialized.")
    return environments[task_name].state()

@app.post("/grader")
def run_grader(req: GraderRequest):
    score = grader.grade(
        req.task_name,
        req.conversation_history,
        req.amount_recovered,
        req.invoice_amount,
        req.steps_taken
    )
    
    if req.invoice_amount > 0:
        ratio = req.amount_recovered / req.invoice_amount
    else:
        ratio = 0
    po = 0.5 if ratio >= 1.0 else (0.3 if ratio >= 0.5 else (0.1 if ratio > 0.0 else 0.0))
    
    neg_score = 0.0
    tone_score = 0.0
    for m in req.conversation_history:
        if m.startswith("Agent:"):
            ml = m.lower()
            neg_score += sum(0.05 for kw in ["value", "relationship", "mutual", "resolve", "commitment", "appreciate", "understand"] if kw in ml)
            tone_score += sum(0.05 for kw in ["appreciate", "understand", "kindly", "request", "please"] if kw in ml)
            tone_score -= sum(0.1 for kw in ["sue", "legal", "useless", "fraud"] if kw in ml)
    
    neg_score = min(neg_score, 0.3)
    tone_score = min(tone_score, 0.2)
    
    explanation = grader.explain_score(score, po, tone_score, neg_score)
    return {"score": score, "explanation": explanation}

@app.post("/baseline")
def run_baseline_endpoint():
    results = {}
    mock_messages = {
        "easy": [
            "We appreciate your business. I am following up on invoice #123.",
            "Can we expect this payment soon? We value you as a client.",
            "Please confirm once the payment is processed. We understand and thank you for being flexible."
        ],
        "medium": [
            "We have noticed your payment is 30 days overdue. We value our relationship.",
            "We understand things take time. Can we arrange a timeline?",
            "Perhaps we can arrange an installment plan or settlement?"
        ],
        "hard": [
            "Hello, we are following up on the invoice #456. We need to clear this balance.",
            "We understand there are concerns regarding the services. We want to be flexible and resolve this mutually.",
            "We are open to a compromise. Would a 40% discount on the remaining balance help us reach an arrangement today?",
            "Let's finalize the settlement at 60% of the original invoice amount. Please confirm."
        ]
    }
    for task_name in TASKS.keys():
        env = PaymentRecoveryEnv(task_name)
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < len(mock_messages[task_name]):
            msg = mock_messages[task_name][steps]
            action = Action(message=msg)
            obs, reward, done, info = env.step(action)
            steps += 1
            
        score = grader.grade(
            task_name,
            obs.conversation_history,
            obs.amount_recovered,
            obs.invoice_amount,
            steps
        )
        results[task_name] = {
            "score": score,
            "steps": steps,
            "recovered": f"${obs.amount_recovered}/${obs.invoice_amount}"
        }
        
    return results
