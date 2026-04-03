from .models import Observation, Action, Reward
from .tasks import TASKS

class PaymentRecoveryEnv:
    def __init__(self, task_name: str):
        if task_name not in TASKS:
            raise ValueError(f"Task {task_name} not found")
        self.task_name = task_name
        self.task_data = TASKS[task_name]
        self.reset()

    def reset(self) -> Observation:
        self.conversation_history = []
        self.amount_recovered = 0.0
        self.current_step = 0
        self.is_done = False
        self.negotiation_stage = 0
        
        return self.state()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.is_done:
            return self.state(), Reward(score=0.0, reason="Episode already done"), True, {}

        self.current_step += 1
        msg = action.message.lower()
        original_msg = action.message
        
        # Repetition detection
        has_repetition = False
        for history_msg in self.conversation_history:
            if history_msg.startswith("Agent:"):
                if original_msg.strip() == history_msg.replace("Agent:", "").strip():
                    has_repetition = True
                    break

        if has_repetition:
            step_reward = -0.2
            client_reply = "You keep repeating yourself."
            self.conversation_history.append(f"Agent: {original_msg}")
            self.conversation_history.append(f"Client: {client_reply}")
            if self.current_step >= 5:
                self.is_done = True
            return self.state(), Reward(score=step_reward, reason="Repetition"), self.is_done, {}

        aggressive_keywords = ["legal", "sue", "court", "fraud", "complaint", "threaten", "report", "useless"]
        empathetic_keywords = ["understand", "flexible", "help", "work together", "resolve", "appreciate", "value"]
        compromise_keywords = ["partial", "installment", "settle", "discount", "arrangement"]
        neutral_keywords = ["reminder", "kindly", "request", "follow up", "invoice", "payment due"]

        is_aggressive = any(kw in msg for kw in aggressive_keywords)
        is_empathetic = any(kw in msg for kw in empathetic_keywords)
        is_compromise = any(kw in msg for kw in compromise_keywords)
        is_neutral = any(kw in msg for kw in neutral_keywords)

        step_reward = 0.0
        client_reply = ""
        done_condition_met = False
        
        status = self.task_data["client_status"]
        invoice_amount = self.task_data["invoice_amount"]

        if is_aggressive:
            step_reward = -0.2
            if status == "disputing":
                client_reply = "I find your tone unacceptable. I will not discuss this further."
                self.amount_recovered = 0.0
                done_condition_met = True
            else:
                client_reply = "There's no need for threats. We are trying to sort this out."

        elif status == "cooperative":
            if is_empathetic or is_neutral or is_compromise:
                if self.negotiation_stage == 0:
                    client_reply = "I understand, I will check and get back to you."
                    self.negotiation_stage = 1
                    step_reward = 0.1
                elif self.negotiation_stage == 1:
                    client_reply = "Okay, I can arrange payment soon."
                    self.negotiation_stage = 2
                    step_reward = 0.15
                elif self.negotiation_stage >= 2:
                    client_reply = "Alright, I will clear the payment."
                    self.amount_recovered = invoice_amount
                    done_condition_met = True
                    step_reward = 0.2
            else:
                client_reply = "I've received your message."

        elif status == "ghosting":
            if is_empathetic or is_compromise or is_neutral:
                if self.negotiation_stage < 2:
                    client_reply = "Will pay next week."
                    self.negotiation_stage += 1
                    step_reward = 0.05
                elif self.negotiation_stage >= 2:
                    client_reply = "Okay, I will pay now."
                    self.amount_recovered = invoice_amount
                    done_condition_met = True
                    step_reward = 0.2
            else:
                client_reply = "I've received your message."

        elif status == "disputing":
            if self.negotiation_stage == 0:
                client_reply = "We are not satisfied with your service."
                step_reward = 0.0
                self.negotiation_stage = 1
            elif self.negotiation_stage == 1:
                if is_empathetic:
                    client_reply = "We understand your point, but still concerned."
                    step_reward = 0.1
                    self.negotiation_stage = 2
                else:
                    client_reply = "We are not going to pay."
                    step_reward = -0.1
            elif self.negotiation_stage == 2:
                if is_compromise:
                    client_reply = "We can consider partial settlement."
                    step_reward = 0.15
                    self.negotiation_stage = 3
                else:
                    client_reply = "This is still not acceptable."
                    step_reward = 0.0
            elif self.negotiation_stage >= 3:
                client_reply = "We agree to settle 60% of the invoice."
                self.amount_recovered = invoice_amount * 0.6
                done_condition_met = True
                step_reward = 0.2

        self.conversation_history.append(f"Agent: {original_msg}")
        self.conversation_history.append(f"Client: {client_reply}")

        if done_condition_met or self.current_step >= 5:
            self.is_done = True
            
        reward = Reward(
            score=step_reward, 
            reason=f"Tone metrics (agg={is_aggressive}, emp={is_empathetic}, comp={is_compromise}, neu={is_neutral})"
        )
        return self.state(), reward, self.is_done, {}

    def state(self) -> Observation:
        return Observation(
            client_name=self.task_data["client_name"],
            invoice_amount=self.task_data["invoice_amount"],
            days_overdue=self.task_data["days_overdue"],
            client_status=self.task_data["client_status"],
            client_personality=self.task_data["client_personality"],
            background=self.task_data["background"],
            conversation_history=self.conversation_history.copy(),
            current_step=self.current_step,
            amount_recovered=self.amount_recovered,
            is_done=self.is_done
        )
