import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def run_baseline():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
        
    client = OpenAI(api_key=api_key)
    
    BASE_URL = "http://127.0.0.1:7860"
    
    system_prompt = (
        "You are a professional payment recovery specialist working "
        "for a digital agency. Your goal is to recover overdue invoice "
        "payments while maintaining client relationships. Be firm but "
        "empathetic. Use logic and professionalism. Avoid legal threats "
        "unless absolute last resort. Reply with ONLY the message you "
        "would send to the client — nothing else."
    )
    
    print("═══════════════════════════════════")
    print("BASELINE RESULTS")
    print("═══════════════════════════════════")

    total_score = 0.0
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        # Reset
        res = requests.post(f"{BASE_URL}/reset", json={"task_name": task})
        obs = res.json()
        
        done = False
        steps = 0
        
        while not done and steps < 5:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Observation: {obs}"}
            ]
            
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            
            agent_msg = completion.choices[0].message.content.strip()
            
            step_res = requests.post(f"{BASE_URL}/step", json={"task_name": task, "message": agent_msg})
            step_data = step_res.json()
            obs = step_data["observation"]
            done = step_data["done"]
            steps += 1
            
        grader_req = {
            "task_name": task,
            "conversation_history": obs["conversation_history"],
            "amount_recovered": obs["amount_recovered"],
            "invoice_amount": obs["invoice_amount"],
            "steps_taken": steps
        }
        
        grader_res = requests.post(f"{BASE_URL}/grader", json=grader_req)
        score_data = grader_res.json()
        score = score_data["score"]
        total_score += score
        
        recovered = obs["amount_recovered"]
        invoice = obs["invoice_amount"]
        
        print(f"Task: {task:<6} | Score: {score:.2f} | Steps: {steps:<5} | Recovered: ${recovered}/${invoice}")
        
    print("───────────────────────────────────")
    print(f"Average Score: {(total_score / len(tasks)):.2f}")
    print("═══════════════════════════════════")

if __name__ == "__main__":
    try:
        run_baseline()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to FastAPI server. Is it running on port 7860?")
