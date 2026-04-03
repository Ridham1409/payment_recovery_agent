import os
import json
from openai import OpenAI
from env.environment import PaymentRecoveryEnv
from env.models import Action
from grader.grader import Grader

def main():
    # Initialize OpenAI client using environment variables
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"]
    )
    model_name = os.environ["payment_recovery_agent"]

    grader = Grader()
    tasks = ["easy", "medium", "hard"]
    
    for task_name in tasks:
        # 1. Reset environment
        env = PaymentRecoveryEnv(task_name)
        obs = env.reset()
        
        # 2. Print [START] log
        start_log = {
            "task": task_name,
            "invoice_amount": obs.invoice_amount,
            "client": obs.client_name
        }
        print(f"[START] {json.dumps(start_log)}")
        
        # 3. Loop max 5 steps
        for step in range(1, 6):
            # Show agent full observation as structured prompt
            prompt = (
                f"Client Name: {obs.client_name}\n"
                f"Invoice Amount: ${obs.invoice_amount}\n"
                f"Days Overdue: {obs.days_overdue}\n"
                f"Client Status: {obs.client_status}\n"
                f"Client Personality: {obs.client_personality}\n"
                f"Background: {obs.background}\n"
                f"Conversation History:\n"
            )
            for msg in obs.conversation_history:
                prompt += f"{msg}\n"
                
            # Agent generates message via OpenAI client
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": agent_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            agent_message = response.choices[0].message.content.strip()
            
            # Call step() with message
            action = Action(message=agent_message)
            obs, reward, is_done, _ = env.step(action)
            
            # Find the client's response from the conversation history (last message usually)
            client_response = ""
            if len(obs.conversation_history) >= 2:
                last_msg = obs.conversation_history[-1]
                if last_msg.startswith("Client:"):
                    client_response = last_msg.replace("Client:", "", 1).strip()
            
            # Print [STEP] log after each step
            step_log = {
                "task": task_name,
                "step": step,
                "agent_message": agent_message,
                "client_response": client_response,
                "reward": float(reward.score),
                "done": is_done
            }
            print(f"[STEP] {json.dumps(step_log)}")
            
            # Break if done=True
            if is_done:
                break
                
        # 4. Call grader with episode data
        final_score = grader.grade(
            task_name=task_name,
            conversation_history=obs.conversation_history,
            amount_recovered=obs.amount_recovered,
            invoice_amount=obs.invoice_amount,
            steps_taken=obs.current_step
        )
        
        # 5. Print [END] log
        end_log = {
            "task": task_name,
            "final_score": float(final_score),
            "amount_recovered": obs.amount_recovered,
            "steps_taken": obs.current_step
        }
        print(f"[END] {json.dumps(end_log)}\n")

if __name__ == "__main__":
    main()
