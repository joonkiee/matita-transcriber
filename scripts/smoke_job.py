#!/usr/bin/env python3
"""Sends a recording through a running Matita and checks the transcript (used by the CI builds).

    python3 scripts/smoke_job.py recording.wav "expected word" [base URL]
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

wav, expect = sys.argv[1], sys.argv[2].lower()
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8765/"


def get(path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.load(r)


for _ in range(120):                                    # wait until the server has started up
    try:
        if not get("api/info").get("starting"):
            break
    except OSError:
        pass
    time.sleep(0.5)

data = open(wav, "rb").read()
req = urllib.request.Request(base + "api/upload", data=data, method="POST", headers={
    "X-Filename": urllib.parse.quote(os.path.basename(wav)), "X-Lang": "en", "X-Quality": "fast",
    "X-Speakers": "off", "X-Transcriber": "1", "Content-Type": "application/octet-stream",
    "Content-Length": str(len(data))})
with urllib.request.urlopen(req, timeout=60) as r:
    jid = json.load(r)["id"]
print("queued", jid)

end, last = time.time() + 600, None
while time.time() < end:
    job = next((j for j in get("api/jobs")["jobs"] if j["id"] == jid), None)
    if job and job["state"] != last:
        last = job["state"]
        print(" ", last, job.get("error") or "")
    if job and job["state"] in ("done", "error", "cancelled"):
        break
    time.sleep(1)
if not job or job["state"] != "done":
    sys.exit("the job didn't finish: %s" % (job and job.get("error")))
text = get("api/text?id=" + jid).get("text", "")
print("transcript:", text.strip()[:200])
if expect not in text.lower():
    sys.exit("expected %r in the transcript" % expect)
print("ok")
