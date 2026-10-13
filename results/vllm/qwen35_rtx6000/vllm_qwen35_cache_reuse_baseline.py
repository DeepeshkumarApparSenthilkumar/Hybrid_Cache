#!/usr/bin/env python3
import csv, json, re, subprocess, time
from pathlib import Path
import requests
from transformers import AutoTokenizer

BASE_URL="http://127.0.0.1:8000"
MODEL="Qwen/Qwen3.5-0.8B"
OUT=Path("results/vllm_qwen35_cache_reuse_baseline.csv")
OUT.parent.mkdir(exist_ok=True)
TARGETS=[1088,2176,4352]
tok=AutoTokenizer.from_pretrained(MODEL)

NAMES={
 "kv":"vllm:kv_cache_usage_perc",
 "q":"vllm:prefix_cache_queries_total",
 "h":"vllm:prefix_cache_hits_total",
 "c":"vllm:prompt_tokens_cached_total",
}

def metric(text,name):
    m=re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$",text,re.M)
    return float(m.group(1)) if m else None

def snapshot():
    t=requests.get(BASE_URL+"/metrics",timeout=15).text
    return {k:metric(t,v) for k,v in NAMES.items()}

def gpu_mib():
    x=subprocess.check_output(
        ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
        text=True).strip().splitlines()[0]
    return float(x)

def prefix(n,tag):
    s=f"{tag}. This is a controlled inference serving benchmark. "
    while len(tok.encode(s,add_special_tokens=False))<n:
        s+="Hybrid inference serving combines attention cache and recurrent state. The same prefix will be reused with a different short question. "
    ids=tok.encode(s,add_special_tokens=False)[:n]
    return tok.decode(ids,skip_special_tokens=True)

def complete(prompt):
    payload={"model":MODEL,"prompt":prompt,"max_tokens":32,"temperature":0,
             "stream":True,"stream_options":{"include_usage":True}}
    t0=time.perf_counter(); first=None; usage={}; out=[]
    with requests.post(BASE_URL+"/v1/completions",json=payload,stream=True,timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "): continue
            data=raw[6:]
            if data=="[DONE]": break
            obj=json.loads(data)
            if obj.get("usage"): usage=obj["usage"]
            for ch in obj.get("choices",[]):
                txt=ch.get("text","")
                if txt:
                    if first is None: first=time.perf_counter()
                    out.append(txt)
    t1=time.perf_counter()
    return {
      "ttft_s": None if first is None else first-t0,
      "e2e_s": t1-t0,
      "prompt_tokens": usage.get("prompt_tokens"),
      "completion_tokens": usage.get("completion_tokens"),
      "text":"".join(out).replace("\n"," ")[:120]
    }

rows=[]
for n in TARGETS:
    pre=prefix(n,f"UNIQUE_CONTEXT_{n}")
    cases=[
      ("cold_first_request", pre+"\n\nQuestion A: What is this benchmark measuring?"),
      ("shared_prefix_reuse", pre+"\n\nQuestion B: Why can prefix reuse reduce prefill work?")
    ]
    for typ,prompt in cases:
        b=snapshot(); gb=gpu_mib()
        r=complete(prompt)
        time.sleep(.5)
        a=snapshot(); ga=gpu_mib()
        dq=None if b["q"] is None or a["q"] is None else a["q"]-b["q"]
        dh=None if b["h"] is None or a["h"] is None else a["h"]-b["h"]
        dc=None if b["c"] is None or a["c"] is None else a["c"]-b["c"]
        hr=None if not dq else dh/dq
        row={
          "target_prefix_tokens":n,"request_type":typ,
          "actual_prompt_tokens":r["prompt_tokens"],"completion_tokens":r["completion_tokens"],
          "ttft_s":r["ttft_s"],"e2e_latency_s":r["e2e_s"],
          "kv_usage_before":b["kv"],"kv_usage_after":a["kv"],
          "prefix_queries_delta_tokens":dq,"prefix_hits_delta_tokens":dh,
          "prefix_hit_rate":hr,"cached_prompt_tokens_delta":dc,
          "gpu_memory_before_mib":gb,"gpu_memory_after_mib":ga,
          "generated_text":r["text"]
        }
        rows.append(row)
        print(row)

with OUT.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("Saved:",OUT)
