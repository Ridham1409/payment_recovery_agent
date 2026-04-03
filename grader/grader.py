class Grader:
    def grade(self, task_name: str, conversation_history: list[str], amount_recovered: float, invoice_amount: float, steps_taken: int) -> float:
        score = 0.0
        
        # Payment Outcome (max 0.5)
        if invoice_amount > 0:
            recovery_ratio = amount_recovered / invoice_amount
        else:
            recovery_ratio = 0.0
            
        payment_outcome_score = 0.0
        if recovery_ratio >= 1.0:
            payment_outcome_score = 0.5
        elif recovery_ratio >= 0.5:
            payment_outcome_score = 0.3
        elif recovery_ratio > 0.0:
            payment_outcome_score = 0.1
        else:
            payment_outcome_score = 0.0
            
        score += payment_outcome_score

        # Negotiation Quality (max 0.3)
        negotiation_keywords = ["value", "relationship", "mutual", "resolve", "commitment", "appreciate", "understand"]
        negotiation_score = 0.0
        
        # Tone Score (max 0.2)
        prof_keywords = ["appreciate", "understand", "kindly", "request", "please"]
        agg_keywords = ["sue", "legal", "useless", "fraud"]
        tone_score = 0.0
        
        for msg in conversation_history:
            if msg.startswith("Agent:"):
                msg_lower = msg.lower()
                for kw in negotiation_keywords:
                    if kw in msg_lower:
                        negotiation_score += 0.05
                for kw in prof_keywords:
                    if kw in msg_lower:
                        tone_score += 0.05
                for kw in agg_keywords:
                    if kw in msg_lower:
                        tone_score -= 0.1
        
        # Caps and constraints
        negotiation_score = min(negotiation_score, 0.3)
        tone_score = min(tone_score, 0.2)
        
        score += negotiation_score + tone_score
        
        # Bonuses
        if steps_taken <= 2 and amount_recovered > 0:
            score += 0.1
        if steps_taken >= 3:
            score += 0.05
        if task_name == "hard" and amount_recovered > 0:
            score += 0.05
            
        return max(0.0, min(1.0, score))

    def explain_score(self, score: float, payment_outcome: float, tone_score: float, negotiation_score: float) -> str:
        return f"Total Score: {score:.2f} | Payment Outcome: {payment_outcome:.2f}/0.50 | Quality: {negotiation_score:.2f}/0.30 | Tone: {tone_score:.2f}/0.20"
