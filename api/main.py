from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, List
from agent.graph import build_graph

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading agent...")
    app_state["agent"] = build_graph()
    print("Agent ready.")
    yield
    app_state.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    equipment: Optional[List[str]] = None
    provider: Optional[str] = None
    api_key: Optional[str] = None

class AnswerResponse(BaseModel):
    intent: str
    answer: str

@app.get("/health")
def health():
    return {"status": "ok", "agent_loaded": "agent" in app_state}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not request.api_key:
        raise HTTPException(
            status_code=400,
            detail="An API key is required. Please provide your own Groq or OpenAI API key."
        )

    try:
        state = {
            "question": request.question,
            "provider": request.provider or "groq",
            "api_key": request.api_key,
        }
        if request.equipment:
            state["equipment"] = request.equipment

        result = app_state["agent"].invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return AnswerResponse(
        intent=result.get("intent", "unknown"),
        answer=result.get("answer", "")
    )
