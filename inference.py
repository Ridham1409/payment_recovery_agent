import os
import sys
import json
from openai import OpenAI
from env.environment import PaymentRecoveryEnv
from env.models import Action
from grader.grader import Grader

SYSTEM_PROMPT = (
    "You are a professional payment recovery specialist working for a digital agency. "
    "Your goal is to recover overdue invoice payments while maintaining client relationships. "
    "Be firm but empathetic. Use logic and professionalism. Avoid legal threats unless absolute last resort. "
    "Reply with ONLY the message you would send to the client — nothing else."
)

def main():
    try:
        api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        hf_token = os.environ.get("HF_TOKEN", "")

        client = OpenAI(
            base_url=api_base_url,
            api_key=hf_token if hf_token else "dummy-key"
        )

        grader = Grader()
        tasks = ["easy", "medium", "hard"]

        for task_name in tasks:
            try:
                env = PaymentRecoveryEnv(task_name)
                obs = env.reset()

                start_log = {
                    "task": task_name,
                    "invoice_amount": obs.invoice_amount,
                    "client": obs.client_name
                }
                print(f"[START] {json.dumps(start_log)}", flush=True)

                for step in range(1, 6):
                    try:
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

                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=512
                        )
                        agent_message = response.choices[0].message.content.strip()

                    except Exception as llm_err:
                        agent_message = "We understand your situation and kindly request you to arrange the payment at your earliest convenience."
                        print(json.dumps({"warning": f"LLM call failed at step {step}: {str(llm_err)}"}), flush=True)

                    action = Action(message=agent_message)
                    obs, reward, is_done, _ = env.step(action)

                    client_response = ""
                    if len(obs.conversation_history) >= 2:
                        last_msg = obs.conversation_history[-1]
                        if last_msg.startswith("Client:"):
                            client_response = last_msg.replace("Client:", "", 1).strip()

                    step_log = {
                        "task": task_name,
                        "step": step,
                        "agent_message": agent_message,
                        "client_response": client_response,
                        "reward": float(reward.score),
                        "done": is_done
                    }
                    print(f"[STEP] {json.dumps(step_log)}", flush=True)

                    if is_done:
                        break

                final_score = grader.grade(
                    task_name=task_name,
                    conversation_history=obs.conversation_history,
                    amount_recovered=obs.amount_recovered,
                    invoice_amount=obs.invoice_amount,
                    steps_taken=obs.current_step
                )

                end_log = {
                    "task": task_name,
                    "final_score": float(final_score),
                    "amount_recovered": obs.amount_recovered,
                    "steps_taken": obs.current_step
                }
                print(f"[END] {json.dumps(end_log)}\n", flush=True)

            except Exception as task_err:
                print(json.dumps({"error": f"Task '{task_name}' failed: {str(task_err)}"}), flush=True)
                continue

    except Exception as e:
        print(json.dumps({"error": str(e), "status": "failed"}), flush=True)
        sys.exit(0)

if __name__ == "__main__":
    main()
