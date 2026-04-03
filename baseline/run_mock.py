import requests
import time

def run_mock_baseline():
    BASE_URL = "http://127.0.0.1:7860"
    
    print("═══════════════════════════════════")
    print("MOCK BASELINE RESULTS (FREE MODE)")
    print("═══════════════════════════════════")

    total_score = 0.0
    tasks = ["easy", "medium", "hard"]
    
    # Updated mock messages with multiple steps to progress the negotiation stage properly
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

    for task in tasks:
        # Reset
        res = requests.post(f"{BASE_URL}/reset", json={"task_name": task})
        obs = res.json()
        
        done = False
        step_idx = 0
        
        while not done and step_idx < len(mock_messages[task]):
            agent_msg = mock_messages[task][step_idx]
            
            step_res = requests.post(f"{BASE_URL}/step", json={"task_name": task, "message": agent_msg})
            step_data = step_res.json()
            obs = step_data["observation"]
            done = step_data["done"]
            step_idx += 1
            
        # Final evaluation via grader
        grader_req = {
            "task_name": task,
            "conversation_history": obs["conversation_history"],
            "amount_recovered": obs["amount_recovered"],
            "invoice_amount": obs["invoice_amount"],
            "steps_taken": step_idx
        }
        
        grader_res = requests.post(f"{BASE_URL}/grader", json=grader_req)
        score_data = grader_res.json()
        score = score_data["score"]
        total_score += score
        
        recovered = obs["amount_recovered"]
        invoice = obs["invoice_amount"]
        
        print(f"Task: {task:<6} | Score: {score:.2f} | Steps: {step_idx:<5} | Recovered: ${recovered}/${invoice}")
        
    print("───────────────────────────────────")
    print(f"Average Score: {(total_score / len(tasks)):.2f}")
    print("═══════════════════════════════════")

if __name__ == "__main__":
    try:
        run_mock_baseline()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to FastAPI server. Run 'python main.py' first.")
