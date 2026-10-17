"""Live training dashboard: python scripts/guard_live.py  ->  http://localhost:8765 (updates every second)."""
import http.server, json, pathlib, re, subprocess, time
BENCH = pathlib.Path(__file__).resolve().parent.parent / "guard_work" / "bench.json"
GB = pathlib.Path(__file__).resolve().parent.parent / "guard_work" / "bench_gpu.json"
LOG = pathlib.Path(__file__).resolve().parent.parent / "guard_work" / "dbg.log"
PAGE = r"""<!doctype html><meta charset=utf-8><title>Verity-Guard training</title>
<style>body{font:15px system-ui;margin:24px;max-width:760px;background:#0f1115;color:#e6e6e6}h1{font-size:20px}
.bar{height:22px;background:#252a33;border-radius:6px;overflow:hidden}.bar>i{display:block;height:100%;background:#4f8cff;width:0;transition:width .5s}
.g{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}.c{background:#181c23;border-radius:8px;padding:10px}.c b{display:block;font-size:20px}
small{color:#8b93a1}pre{background:#181c23;padding:10px;border-radius:8px;height:200px;overflow:auto;font-size:12px}canvas{width:100%;height:150px;background:#181c23;border-radius:8px}</style>
<h1>Verity-Guard: fine-tuning on GPU</h1><div class=bar><i id=bar></i></div><p id=st></p>
<div class=g><div class=c><small>Step</small><b id=step>-</b></div><div class=c><small>Loss</small><b id=loss>-</b></div>
<div class=c><small>Elapsed / ETA</small><b id=eta>-</b></div><div class=c><small>GPU</small><b id=gpu>-</b></div></div>
<canvas id=cv width=720 height=150></canvas><h3>Live benchmark (int8, CPU, test set)</h3><div class=bar><i id=bb></i></div><pre id=bench>not started</pre><h3>Live benchmark (fp16, GPU, test set)</h3><div class=bar><i id=gb></i></div><pre id=gbench>not started</pre><h3>Validation (per epoch)</h3><pre id=val>waiting...</pre><h3>Log</h3><pre id=log></pre>
<script>const $=i=>document.getElementById(i);async function t(){try{const d=await (await fetch('/status')).json();
$('bar').style.width=d.pct+'%';$('st').textContent=d.state+' - '+d.pct.toFixed(1)+'%';$('step').textContent=d.step+' / '+d.total;$('loss').textContent=d.loss??'-';
$('eta').textContent=d.elapsed+'s / '+(d.eta??'-')+'s';$('gpu').textContent=d.gpu;$('val').textContent=d.val.join('\n')||'waiting for epoch 1 to finish...';$('log').textContent=d.tail.join('\n');$('log').scrollTop=1e9;
const c=$('cv').getContext('2d');c.clearRect(0,0,720,150);const ls=d.losses;if(ls.length>1){const m=Math.max(...ls);c.strokeStyle='#4f8cff';c.beginPath();ls.forEach((v,i)=>{const x=i/(ls.length-1)*710+5,y=145-v/m*135;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()}}catch(e){}}
async function b(){try{const d=await (await fetch('/bench')).json();$('bb').style.width=(100*d.done/d.total)+'%';$('bench').textContent=`${d.done}/${d.total} examples   ${d.per_s}/s   ETA ${d.eta_s}s${d.finished?'   FINISHED':''}
running  acc ${d.acc}  precision ${d.precision}  recall ${d.recall}  F1 ${d.f1}
latency  p50 ${d.p50} ms   p95 ${d.p95} ms`}catch(e){}}
async function gg(){try{const d=await (await fetch('/bench_gpu')).json();$('gb').style.width=(100*d.done/d.total)+'%';$('gbench').textContent=`${d.phase}: ${d.done}/${d.total}   ${d.per_s}/s${d.finished?'   FINISHED':''}
running  acc ${d.acc}  F1 ${d.f1}
latency  p50 ${d.p50} ms   p95 ${d.p95} ms`}catch(e){}}
setInterval(t,1000);setInterval(b,1000);setInterval(gg,1000);t();b();gg()</script>"""
def status():
    txt = LOG.read_text(errors="ignore") if LOG.exists() else ""
    lines = [l for l in txt.splitlines() if "symlink" not in l.lower() and "warn" not in l.lower() and l.strip()]
    st = re.findall(r"ep(\d+) step (\d+)/(\d+) loss ([\d.]+) (\d+)s", txt)
    last = st[-1] if st else None
    step, total = (int(last[1]), int(last[2])) if last else (0, 1)
    el = int(last[4]) if last else 0
    try: g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception: g = "n/a"
    done = "DONE" in txt
    return {"pct": 100.0 if done else 100 * step / total, "step": step, "total": total, "loss": last[3] if last else None, "elapsed": el,
            "eta": int(el / step * (total - step)) if step else None, "gpu": g, "state": "finished" if done else ("training" if last else "loading model / tokenizing"),
            "val": [l for l in lines if l.startswith("EPOCH") or l.startswith("DONE")], "tail": lines[-14:], "losses": [float(x[3]) for x in st][-200:]}
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(s):
        b, ct = (json.dumps(status()).encode(), "application/json") if s.path == "/status" else ((BENCH.read_bytes() if BENCH.exists() else b"{}"), "application/json") if s.path == "/bench" else ((GB.read_bytes() if GB.exists() else b"{}"), "application/json") if s.path == "/bench_gpu" else (PAGE.encode(), "text/html")
        s.send_response(200); s.send_header("Content-Type", ct); s.send_header("Cache-Control", "no-store"); s.end_headers(); s.wfile.write(b)
    def log_message(s, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
