from typing import TypedDict, NotRequired, Optional, List

class AgentState(TypedDict):
    question: str
    equipment: NotRequired[Optional[List[str]]]
    intent: NotRequired[str]        # "exercise_lookup" | "program_request" | "conversational" | "off_topic"
    context: NotRequired[Optional[str]]
    answer: NotRequired[Optional[str]]
