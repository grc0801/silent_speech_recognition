import subprocess
import sys

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


def speak(text):
    if pyttsx3 is None:
        return
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


proc = subprocess.Popen(
    [sys.executable, r"D:\lip_tracking_project\demo\predict_live.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in proc.stdout:
    print(line, end="")
    if "FINISHED!" in line:
        word = line.split("FINISHED!", 1)[-1].strip()
        if word:
            speak(word)

proc.wait()
sys.exit(proc.returncode)