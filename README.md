# AI Payment Recovery & Negotiation Environment

This environment simulates real-world invoice recovery negotiation, where AI agents must handle delays, resistance, and disputes through multi-step reasoning and strategic communication.

## 🎯 Why this is hard for AI

Unlike simple chat tasks, this environment requires:
- **Multi-turn reasoning**
- **Emotional intelligence**
- **Negotiation strategy**
- **Dynamic decision making**

An agent must be able to read the room, understand when to be firm versus empathetic, and propose strategic compromises when a client is heavily disputing an invoice.

## 1. Overview
Simulate a real-world system where an AI agent acts as a professional accounts manager trying to recover overdue invoice payments from clients. The environment rigorously evaluates reasoning ability, negotiation quality, communication tone, and the ultimate payment recovery outcome using a dynamic progression state machine.

## 2. Real-world Motivation
In B2B scenarios, accounts receivable represents significant working capital. Collecting these efficiently via AI saves time, but the AI must preserve customer relationships. A binary outcome is unrealistic—resistance, hesitation, and partial settlements are common. Tone, empathy, and negotiation skills are paramount to recovering funds without alienating the client.

## 3. Tasks Table
| ID | Description | Difficulty | Context |
|---|---|---|---|
| `easy` | Recover $1000 from cooperative client, 7 days overdue | Easy | Cooperative but forgetful. Needs consistent, polite follow-up. |
| `medium` | Recover $3000 from ghosting client, 30 days overdue | Medium | Avoidant, reads messages but delays. Needs strategic escalation. |
| `hard` | Negotiate disputed $8000 invoice, 90 days overdue | Hard | Aggressive, claims work quality was poor. Requires deep reasoning and compromise. |

## 💬 4. Sample Conversation (Hard Difficulty / Disputing Client)

**Step 1:**
- **AI Agent**: *Hello, we are following up on the invoice #456. We need to clear this balance.*
- **Client Simulator**: *We are not satisfied with your service.*

**Step 2:**
- **AI Agent**: *We understand there are concerns regarding the services. We want to be flexible and resolve this mutually.*
- **Client Simulator**: *We understand your point, but still concerned.*

**Step 3:**
- **AI Agent**: *We are open to a compromise. Would a 40% discount on the remaining balance help us reach an arrangement today?*
- **Client Simulator**: *We can consider partial settlement.*

**Step 4:**
- **AI Agent**: *Let's finalize the settlement at 60% of the original invoice amount. Please confirm.*
- **Client Simulator**: *We agree to settle 60% of the invoice.*

*(Outcome: $4,800 recovered, 4 steps taken, deep negotiation verified)*

## 5. Observation & Action Space

**Observation Space**:
- `client_name`: (str) Name of the corporate client
- `invoice_amount`: (float) Original value of the invoice
- `days_overdue`: (int) Days past due date
- `client_status`: (str) Current behavior: cooperative, ghosting, disputing
- `client_personality`: (str) Client's communication style
- `background`: (str) Deep background about historical dealings
- `conversation_history`: (List[str]) Log of messages
- `current_step`: (int) Turn counter (max 5)
- `amount_recovered`: (float) Running total of recovered funds
- `is_done`: (bool) End of episode marker

**Action Space**:
Language-based action space:
```python
class Action(BaseModel):
    message: str # Natural language message
```

## 6. Reward Function & Grader
- **Step-wise Rewards**: Penalties for aggressive tone (-0.2), rewards for empathetic (+0.2) or compromise (+0.15) tones. Penalties for message repetition (-0.2).
- **Grader Breakdown**:
  - Payment Outcome: Max 0.5 points (Proportional to recovery ratio)
  - Negotiation Quality: Max 0.3 points (Keywords mapping)
  - Tone Score: Max 0.2 points (Professionalism vs Aggression)
  - Bonuses: For deeper reasoning in hard scenarios (>= 3 steps) or fast resolution for easy tasks.

## 7. Customer Simulator Logic
State-machine based simulation engine:
- Tracks `negotiation_stage` across all interactions.
- Scans agent messages for thematic keyword groups (aggressive, empathetic, compromise, neutral).
- Different client personas react contextually to the tone and stage, prohibiting one-shot generic recoveries.
- Tracks done conditions like full payment, partial agreement, hard refusal, or step limit.

## 8. Installation
```bash
pip install -r requirements.txt
```

## 9. Running APIs and Baselines
Start the API locally:
```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```
Then run the baseline test (ensure OPENAI_API_KEY is in your `.env`):
```bash
python baseline/run.py
```
Or run the deterministic free mock baseline:
```bash
python baseline/run_mock.py
```

## 10. API Endpoints Table (OpenEnv Compliant)
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Check API status |
| GET | `/tasks` | List all available tasks with full schema detail |
| POST | `/reset` | Initializes a new environment session |
| POST | `/step` | Execute agent interaction against the simulator |
| GET | `/state` | Retrieve current state without stepping |
| POST | `/grader` | Evaluate episode performance manually |
| POST | `/baseline` | Return evaluation results for the internal mock AI across tasks |

## 11. Custom Baseline Scores
| Task | Difficulty | Score | Steps Taken | Recovery Outcome |
|---|---|---|---|---|
| `easy` | Easy | 0.85 | 3 | $1,000 / $1,000 |
| `medium` | Medium | 0.75 | 3 | $3,000 / $3,000 |
| `hard` | Hard | 0.65 | 4 | $4,800 / $8,000 (Partial allowed) |

## 12. Author
Ridham Bhavnagariya, Xpeartz

## 13. Hugging Face Deployment

This project is built to be easily deployed to Hugging Face Spaces using Docker.

- **Space SDK**: Docker
- **Port**: 7860
- **Tag**: openenv

### Required Environment Variables
When setting up your space, ensure the following environment variables are added in the Space's settings:
- `API_BASE_URL` - The endpoint for your LLM API
- `MODEL_NAME` - The specific model identifier to be used
- `HF_TOKEN` - Your Hugging Face API key or required token

The space will properly respond to a `GET /health` with `200 OK` indicating the FastAPI server is running successfully on the required port.
