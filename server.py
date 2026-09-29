# AI Agent web server (Phase 1 deployment).  
# Exposes the agent over HTTP with a chat UI at http://127.0.0.1:8000.  
# Conversations persist per session under data/sessions/.  
  
import argparse  
import asyncio  
import logging  
import time  
import uuid  
from pathlib import Path  
  
from fastapi import FastAPI, HTTPException  
from fastapi.responses import FileResponse  
from pydantic import BaseModel, Field  
  
from core.agent import Agent, AgentMode
from core.session_store import SessionStore
from core.memory import Memory
from tools import get_all_tools

  
HERE = Path(__file__).resolve().parent  
LOGS_DIR = HERE / 'logs'  
LOGS_DIR.mkdir(parents=True, exist_ok=True)  
  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',  
    handlers=[  
        logging.FileHandler(LOGS_DIR / 'ai-agent.log', encoding='utf-8'),  
        logging.StreamHandler(),  
    ],  
)  
logger = logging.getLogger('ai-agent')  
  
store = SessionStore()  
agents = {}  
locks = {}  
last_used = {}  
MAX_CACHED_AGENTS = 100  
  
  
class ChatRequest(BaseModel):  
    session_id: str = Field(default='', description='Existing session id, or empty for a new chat.')  
    message: str = Field(default='', description='User message.')  
    mode: str = Field(default='react', description='react, plan or direct.')  
  
  
class RememberRequest(BaseModel):  
    content: str = Field(default='', description='Fact to remember.')  
  
  
def _build_agent(session_id: str) -> Agent:  
    agent = Agent(  
        mode=AgentMode.REACT,  
        tools=get_all_tools(),  
        max_iterations=10,  
        verbose=False,  
        auto_memory=True,  
    )  
    data = store.load(session_id)  
    if data and data.get('messages'):  
        agent.memory.conversation.load_messages(data['messages'])  
    return agent  
  
  
def _get_agent(session_id: str) -> Agent:  
    agent = agents.get(session_id)  
    if agent is None:  
        agent = _build_agent(session_id)  
        agents[session_id] = agent  
        locks[session_id] = asyncio.Lock()  
        _evict_if_needed()  
    last_used[session_id] = time.time()  
    return agent  
  
  
def _evict_if_needed() -> None:  
    while len(agents) > MAX_CACHED_AGENTS:  
        oldest = min(agents, key=lambda sid: last_used.get(sid, 0))  
        del agents[oldest]  
        locks.pop(oldest, None)  
        last_used.pop(oldest, None)
  
  
def _session_title(session_id: str, message: str) -> str:  
    data = store.load(session_id)  
    if data and data.get('title'):  
        return data['title']  
    text = ' '.join(message.split())  
    return text[:24] if text else 'New chat'  
  
  
app = FastAPI(title='AI Agent', version='1.0.0')  
  
  
@app.get('/')  
async def index():  
    return FileResponse(HERE / 'static' / 'index.html')  
  
  
@app.get('/api/health')  
async def health():  
    return {'status': 'ok', 'sessions': len(agents), 'time': time.time()}  
  
  
@app.post('/api/chat')  
async def chat(req: ChatRequest):  
    message = req.message.strip()  
    if not message:  
        raise HTTPException(status_code=400, detail='Message is empty.')  
    session_id = req.session_id.strip() or uuid.uuid4().hex[:12]  
    mode_name = req.mode.strip().lower() or 'react'  
    mode_map = {  
        'react': AgentMode.REACT,  
        'plan': AgentMode.PLAN_AND_EXECUTE,  
        'direct': AgentMode.DIRECT,  
    }  
    if mode_name not in mode_map:  
        raise HTTPException(status_code=400, detail='Unknown mode. Use react, plan or direct.')  
  
    agent = _get_agent(session_id)  
    async with locks[session_id]:  
        logger.info('[%s] user: %s', session_id, message[:200])  
        started = time.time()  
        agent.mode = mode_map[mode_name]  
        reply = await agent.run(message)  
        elapsed = time.time() - started  
        logger.info('[%s] agent: %s (%.1fs)', session_id, reply[:200], elapsed)  
  
    title = _session_title(session_id, message)  
    store.save(session_id, agent.memory.conversation.to_dict(), title)  
    return {  
        'session_id': session_id,  
        'reply': reply,  
        'mode': mode_name,  
    }  
  
  
@app.get('/api/sessions')  
async def sessions():  
    return store.list_sessions()  
  
  
@app.get('/api/sessions/{session_id}')  
async def session_detail(session_id: str):  
    data = store.load(session_id)  
    if data is None:  
        raise HTTPException(status_code=404, detail='Session not found.')  
    return data  
  
  
@app.delete('/api/sessions/{session_id}')  
async def delete_session(session_id: str):  
    agents.pop(session_id, None)  
    locks.pop(session_id, None)  
    last_used.pop(session_id, None)  
    removed = store.delete(session_id)  
    if not removed:  
        raise HTTPException(status_code=404, detail='Session not found.')  
    return {'deleted': session_id}  
  
  
@app.get('/api/memories')  
async def memories():  
    mem = Memory()  
    if mem.long_term is None:  
        return {'memories': []}  
    return {'memories': mem.long_term.get_all()}  
  
  
@app.post('/api/memories')  
async def remember(req: RememberRequest):  
    mem = Memory()  
    ok = mem.store_fact(req.content)  
    return {'stored': ok}  
  
  
@app.delete('/api/memories/{target}')  
async def forget(target: str):  
    mem = Memory()  
    removed = mem.forget(target)  
    return {'removed': removed}  
  
  
def main() -> None:
    parser = argparse.ArgumentParser(description='AI Agent web server')  
    parser.add_argument('--host', default='127.0.0.1', help='Bind address (0.0.0.0 for LAN access).')  
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on.')  
    args = parser.parse_args()  
  
    import uvicorn  
  
    logger.info('Starting AI Agent server at http://%s:%s', args.host, args.port)  
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')  
  
  
if __name__ == '__main__':  
    main()  
