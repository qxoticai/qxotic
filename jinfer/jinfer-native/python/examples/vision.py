"""Vision: an image is just a value next to the question. `python vision.py photo.jpg`"""
import sys, urllib.request, jinfer
from pathlib import Path

if len(sys.argv) > 1:
    picture = Path(sys.argv[1])
else:  # the windmills of Consuegra, La Mancha: Don Quixote's "giants"
    req = urllib.request.Request("https://qxotic.ai/snippets/consuegra.jpg", headers={"User-Agent": "jinfer-example"})
    picture = urllib.request.urlopen(req).read()           # bytes work too

with jinfer.load("LiquidAI/LFM2.5-VL-450M-GGUF:Q8_0",
                  companions={"media": "LiquidAI/LFM2.5-VL-450M-GGUF/mmproj-LFM2.5-VL-450m-Q8_0.gguf"}) as vl:
    reply = vl.chat([picture, "Don Quixote saw giants here. What do you see? One sentence."],
                    max_tokens=60, temperature=0, reasoning_effort="none")
    print(reply.text)
