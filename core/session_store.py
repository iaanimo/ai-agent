# Session persistence for the web server.  
# Each chat session keeps its own conversation history on disk under  
# data/sessions/<session_id>.json so the assistant survives restarts.  
  
import json  
import time  
from pathlib import Path  
from typing import Optional  
  
from config.settings import get_settings  
  
SESSION_FILE_VERSION = 1  
  
  
class SessionStore:  
    # Load and save per-session conversation history.  
  
    def __init__(self, sessions_dir: Optional[Path] = None):  
        root = Path(sessions_dir) if sessions_dir else (get_settings().data_dir / 'sessions')  
        self.sessions_dir = root  
        self.sessions_dir.mkdir(parents=True, exist_ok=True)  
  
    def _path(self, session_id: str) -> Path:  
        safe = ''.join(c for c in session_id if c.isalnum() or c in '-_')  
        return self.sessions_dir / (safe + '.json') 
  
    def save(self, session_id: str, messages: list, title: str = '') -> None:  
        path = self._path(session_id)  
        old = self.load(session_id)  
        created = old['created_at'] if old else time.time()  
        data = {  
            'version': SESSION_FILE_VERSION,  
            'session_id': session_id,  
            'title': title,  
            'created_at': created,  
            'updated_at': time.time(),  
            'messages': messages,  
        }  
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')  
  
    def load(self, session_id: str) -> Optional[dict]:  
        path = self._path(session_id)  
        if not path.exists():  
            return None  
        try:  
            return json.loads(path.read_text(encoding='utf-8'))  
        except (json.JSONDecodeError, OSError):  
            return None 
  
    def list_sessions(self) -> list:  
        sessions = []  
        for path in self.sessions_dir.glob('*.json'):  
            data = self.load(path.stem)  
            if data is None:  
                continue  
            sessions.append({  
                'session_id': data.get('session_id', path.stem),  
                'title': data.get('title', ''),  
                'created_at': data.get('created_at', 0),  
                'updated_at': data.get('updated_at', 0),  
                'message_count': len(data.get('messages', [])),  
            })  
        sessions.sort(key=lambda item: item['updated_at'], reverse=True)  
        return sessions  
  
    def delete(self, session_id: str) -> bool:  
        path = self._path(session_id)  
        if path.exists():  
            path.unlink()  
            return True  
        return False 
