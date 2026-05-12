from app.config import AgentState

def node_router(state: AgentState):
    if state['next_action'] == "retrieve":
        return 'retrieve'
    return 'final_answer'
