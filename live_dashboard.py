#!/usr/bin/env python3
import json, re, subprocess, threading, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HOST, PORT, TARGET, INTERVAL, PING_COUNT = "127.0.0.1", 8765, "8.8.8.8", 30, 5
OUT = Path("results/live_telemetry.jsonl")
history, lock, gateway = [], threading.Lock(), None

def get_gateway():
    try:
        p = subprocess.run(["route","-n","get","default"], capture_output=True, text=True, timeout=5)
        m = re.search(r"gateway:\s*(\S+)", p.stdout)
        return m.group(1) if m else None
    except (OSError, subprocess.TimeoutExpired):
        return None

def ping_host(host):
    empty = {"reachable":False,"packet_loss_pct":None,"min_ms":None,"avg_ms":None,"max_ms":None}
    if not host: return empty
    try:
        p = subprocess.run(["ping","-c",str(PING_COUNT),host],capture_output=True,text=True,timeout=20)
        out = p.stdout
        loss = re.search(r"([\d.]+)% packet loss",out)
        rtt = re.search(r"(?:round-trip|rtt) min/avg/max/(?:stddev|mdev) = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)",out)
        lp = float(loss.group(1)) if loss else None
        return {"reachable":lp is not None and lp < 100,"packet_loss_pct":lp,
                "min_ms":float(rtt.group(1)) if rtt else None,
                "avg_ms":float(rtt.group(2)) if rtt else None,
                "max_ms":float(rtt.group(3)) if rtt else None}
    except (OSError, subprocess.TimeoutExpired): return empty

def collector():
    global gateway
    gateway = get_gateway()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    while True:
        row = {"timestamp":datetime.now().astimezone().isoformat(timespec="seconds"),
               "gateway":gateway,"gateway_ping":ping_host(gateway),"internet_ping":ping_host(TARGET),
               "source":"live ping measurements","note":"Measured telemetry, not a GNN prediction."}
        with lock:
            history.append(row)
            del history[:-120]
        try:
            with OUT.open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
        except OSError as e: print("Could not save telemetry:",e)
        print(row["timestamp"],"gateway avg:",row["gateway_ping"]["avg_ms"],"ms; internet avg:",row["internet_ping"]["avg_ms"],"ms")
        time.sleep(INTERVAL)

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Live Network Health</title>
<style>
body{font-family:system-ui;background:#10151d;color:#e8eef7;margin:0;padding:24px}.wrap{max-width:1100px;margin:auto}.sub,.small{color:#9caec4}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}.card{background:#1b2532;border:1px solid #304154;border-radius:14px;padding:18px}.value{font-size:30px;font-weight:700;margin:8px 0}.pill{display:inline-block;padding:7px 10px;border-radius:20px;background:#213c31;color:#8ee5b0}.warn{background:#49351d;color:#ffd18b}.notice{border-left:4px solid #e3ad55;padding:12px;background:#28271f;margin:16px 0}canvas{width:100%;height:220px;background:#141d28;border-radius:10px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #304154;font-size:13px}
</style></head><body><div class="wrap"><h1>Live Network Health Monitor</h1><p class="sub">Actual measurements from this Mac • refreshes every 10 sec • collection every 30 sec</p>
<div class="notice"><b>Scope:</b> This dashboard measures gateway and internet reachability, latency and packet loss. It does not claim GNN predictions or full-router topology monitoring.</div><div id="status" class="pill">Waiting for first sample…</div>
<div class="grid" style="margin-top:14px"><div class="card"><div class="sub">Internet average latency</div><div id="ia" class="value">—</div><div id="il" class="small">Packet loss: —</div></div>
<div class="card"><div class="sub">Gateway average latency</div><div id="ga" class="value">—</div><div id="gl" class="small">Packet loss: —</div></div>
<div class="card"><div class="sub">Gateway</div><div id="gw" class="value" style="font-size:23px">—</div><div class="small">Default route detected by macOS</div></div>
<div class="card"><div class="sub">Last sample</div><div id="ts" class="value" style="font-size:18px">—</div><div id="count" class="small">Samples: 0</div></div></div>
<div class="card" style="margin-top:14px"><h3>Recent average latency (ms)</h3><canvas id="chart" width="1000" height="220"></canvas><div class="small">Gateway = amber • Internet = blue</div></div>
<div class="card" style="margin-top:14px"><h3>Recent samples</h3><table><thead><tr><th>Time</th><th>Gateway avg / loss</th><th>Internet avg / loss</th><th>Status</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="small">Saved to results/live_telemetry.jsonl • local-only server</p></div>
<script>
async function refresh(){try{let d=await(await fetch('/api')).json(),a=d.history||[],x=a[a.length-1];if(!x){document.getElementById('status').textContent='Waiting for first ping cycle (usually 10–25 sec)…';return}
let g=x.gateway_ping||{},n=x.internet_ping||{},fmt=v=>v==null?'—':Number(v).toFixed(1)+' ms',loss=v=>v==null?'—':Number(v).toFixed(1)+'%';
document.getElementById('ia').textContent=fmt(n.avg_ms);document.getElementById('il').textContent='Packet loss: '+loss(n.packet_loss_pct);document.getElementById('ga').textContent=fmt(g.avg_ms);document.getElementById('gl').textContent='Packet loss: '+loss(g.packet_loss_pct);document.getElementById('gw').textContent=x.gateway||'Not found';document.getElementById('ts').textContent=x.timestamp;document.getElementById('count').textContent='Samples this session: '+a.length;
let alerts=[];if(n.packet_loss_pct>0)alerts.push('Internet packet loss');if(g.packet_loss_pct>0)alerts.push('Gateway packet loss');if(n.avg_ms>100)alerts.push('Internet latency >100 ms');if(g.avg_ms>20)alerts.push('Gateway latency >20 ms');let s=document.getElementById('status');s.textContent=alerts.length?'Warning: '+alerts.join(' • '):'No threshold alerts in latest sample';s.className='pill '+(alerts.length?'warn':'');
document.getElementById('rows').innerHTML=a.slice(-8).reverse().map(q=>{let z=q.gateway_ping||{},y=q.internet_ping||{};let st=(y.packet_loss_pct>0||z.packet_loss_pct>0||y.avg_ms>100||z.avg_ms>20)?'Check':'No threshold alert';return `<tr><td>${q.timestamp}</td><td>${fmt(z.avg_ms)} / ${loss(z.packet_loss_pct)}</td><td>${fmt(y.avg_ms)} / ${loss(y.packet_loss_pct)}</td><td>${st}</td></tr>`}).join('');draw(a.slice(-30))
}catch(e){document.getElementById('status').textContent='Starting dashboard…'}}
function draw(a){let c=document.getElementById('chart'),x=c.getContext('2d'),w=c.width,h=c.height;x.clearRect(0,0,w,h);x.strokeStyle='#344355';for(let i=1;i<5;i++){x.beginPath();x.moveTo(35,i*h/5);x.lineTo(w-12,i*h/5);x.stroke()}let v=a.flatMap(q=>[(q.gateway_ping||{}).avg_ms,(q.internet_ping||{}).avg_ms]).filter(q=>q!=null),m=Math.max(100,...v)*1.1;function line(k,color){x.beginPath();x.strokeStyle=color;x.lineWidth=2;let started=false;a.forEach((q,i)=>{let n=(q[k]||{}).avg_ms;if(n==null)return;let px=35+i*(w-50)/Math.max(1,a.length-1),py=h-20-n/m*(h-35);if(!started){x.moveTo(px,py);started=true}else x.lineTo(px,py)});x.stroke()}line('gateway_ping','#f0b45b');line('internet_ping','#66b6ff');x.fillStyle='#9caec4';x.fillText('0 ms',4,h-18);x.fillText(Math.round(m)+' ms',2,15)}refresh();setInterval(refresh,10000)
</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path == "/api":
            with lock: body=json.dumps({"history":history[-120:]})
            self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(body.encode())
        elif urlparse(self.path).path == "/":
            body=HTML.encode();self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
        else:self.send_error(404)
    def log_message(self,*args):pass

def main():
    threading.Thread(target=collector,daemon=True).start()
    server=ThreadingHTTPServer((HOST,PORT),Handler)
    print(f"Dashboard: http://{HOST}:{PORT}")
    print("Actual ping telemetry every 30 seconds. Stop with Ctrl+C.")
    try:server.serve_forever()
    except KeyboardInterrupt:print("\\nDashboard stopped.")
    finally:server.server_close()
if __name__=="__main__":main()
