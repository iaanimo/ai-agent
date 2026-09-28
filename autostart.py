# Register the AI Agent server to start automatically at logon.  
# Usage:  
#   .venv\Scripts\python.exe autostart.py install  
#   .venv\Scripts\python.exe autostart.py uninstall  
#   .venv\Scripts\python.exe autostart.py status  
  
import subprocess  
import sys  
from pathlib import Path  
  
HERE = Path(__file__).resolve().parent  
TASK_NAME = 'AI Agent Server'  
START_BAT = HERE / 'start.bat'  
Q = chr(34)  
TR = 'cmd /c start /min ' + Q + Q + ' ' + Q + str(START_BAT) + Q  
  
  
def run(args):  
    return subprocess.run(args, capture_output=True, text=True)  
  
  
def install():  
    result = run(['schtasks', '/Create', '/TN', TASK_NAME, '/TR', TR, '/SC', 'ONLOGON', '/RL', 'LIMITED', '/F'])  
    print(result.stdout)  
    print(result.stderr)  
    return result.returncode  
  
  
def uninstall():  
    result = run(['schtasks', '/Delete', '/TN', TASK_NAME, '/F'])  
    print(result.stdout)  
    print(result.stderr)  
    return result.returncode  
  
  
def status():  
    result = run(['schtasks', '/Query', '/TN', TASK_NAME])  
    print(result.stdout)  
    print(result.stderr)  
    return 0  
  
  
def main():  
    action = sys.argv[1] if len(sys.argv) > 1 else 'status'  
    if action == 'install':  
        sys.exit(install())  
    elif action == 'uninstall':  
        sys.exit(uninstall())  
    else:  
        sys.exit(status())  
  
  
if __name__ == '__main__':  
    main()  
