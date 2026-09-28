# Tests for SessionStore persistence.  
  
from core.session_store import SessionStore  
  
  
def test_save_and_load_roundtrip(tmp_path):  
    store = SessionStore(sessions_dir=tmp_path)  
    store.save('abc123', [{'role': 'human', 'content': 'hi'}, {'role': 'ai', 'content': 'hello'}], title='Greeting')  
    data = store.load('abc123')  
    assert data is not None  
    assert data['title'] == 'Greeting'  
    assert len(data['messages']) == 2  
    assert data['messages'][0]['role'] == 'human'  
  
  
def test_load_missing_returns_none(tmp_path):  
    store = SessionStore(sessions_dir=tmp_path)  
    assert store.load('nope') is None  
  
  
def test_list_sessions(tmp_path):  
    store = SessionStore(sessions_dir=tmp_path)  
    store.save('first', [{'role': 'human', 'content': 'a'}], title='A')  
    store.save('second', [{'role': 'human', 'content': 'b'}], title='B')  
    sessions = store.list_sessions()  
    assert len(sessions) == 2  
  
  
def test_delete_session(tmp_path):  
    store = SessionStore(sessions_dir=tmp_path)  
    store.save('one', [{'role': 'human', 'content': 'x'}], title='One')  
    assert store.delete('one') is True  
    assert store.load('one') is None  
    assert store.delete('one') is False  
