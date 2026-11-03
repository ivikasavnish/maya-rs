"""Line-oriented text normaliser for maya-rs: one JSON request per stdin line {"text", "lang"} -> {"out"} or {"error"}.
Pure standard library (the maya3.text normaliser, copied from maya2-native c6b37686)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maya3.text import normalise_text  # noqa: E402

normalise_text("1", "en")
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        out = {"out": normalise_text(req["text"], req["lang"])}
    except Exception as exc:  # never kill the bridge on one bad request
        out = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    sys.stdout.flush()
