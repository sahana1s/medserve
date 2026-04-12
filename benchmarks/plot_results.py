"""
benchmarks/plot_results.py — Generate all 5 paper figures from saved JSON logs.
Run after all three benchmark modes have completed.
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                     "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})

COLORS = {"fifo":"#888780","static_batch":"#BA7517","triton":"#378ADD","medserve":"#1D9E75"}
LABELS = {"fifo":"FIFO (baseline)","static_batch":"Static batching",
          "triton":"Triton","medserve":"MedServe (ours)"}
LOAD_LEVELS = ["low","medium","high"]
LOAD_RATES  = {"low":5,"medium":20,"high":50}
SLA         = {"icu":100,"nlp":300,"imaging":500}

def _load(path):
    with open(path) as f: return json.load(f)["records"]

def _lats(recs, mt=None):
    if mt: recs=[r for r in recs if r["model_type"]==mt]
    return sorted(r["total_latency_ms"] for r in recs)

def _viol(recs, mt=None):
    if mt: recs=[r for r in recs if r["model_type"]==mt]
    return 100*sum(1 for r in recs if r["sla_violated"])/max(len(recs),1)

def _pct(v,p): return v[max(0,int(p*len(v))-1)] if v else 0

def _exists(log_dir, mode, level):
    return (Path(log_dir)/f"{mode}_{level}.json").exists()

# Figure 1: CDF
def fig_cdf(log_dir, level="medium", out_dir="results/plots"):
    modes=["fifo","static_batch","triton","medserve"]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,mt in zip(axes,["icu","nlp","imaging"]):
        for mode in modes:
            p=Path(log_dir)/f"{mode}_{level}.json"
            if not p.exists(): continue
            lats=_lats(_load(str(p)),mt)
            if not lats: continue
            ax.plot(lats,np.linspace(0,1,len(lats)),color=COLORS[mode],linewidth=2,
                    linestyle="--" if mode in("fifo","static_batch") else "-",label=LABELS[mode])
        ax.axvline(SLA[mt],color="#E24B4A",linewidth=1.2,linestyle=":",label=f"SLA {SLA[mt]}ms")
        ax.set_title(f"{mt.upper()}"); ax.set_xlabel("Latency (ms)"); ax.grid(alpha=.3)
        if mt=="icu": ax.set_ylabel("CDF")
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=5,bbox_to_anchor=(.5,-.08),fontsize=8)
    fig.suptitle(f"Latency CDF — {level} load"); fig.tight_layout()
    out=f"{out_dir}/fig1_cdf_{level}.png"; fig.savefig(out,bbox_inches="tight"); plt.close()
    print(f"  {out}")

# Figure 2: P99 vs load
def fig_p99_load(log_dir, mt="icu", out_dir="results/plots"):
    fig,ax=plt.subplots(figsize=(6,4))
    for mode in ["fifo","static_batch","triton","medserve"]:
        xs,ys=[],[]
        for level in LOAD_LEVELS:
            p=Path(log_dir)/f"{mode}_{level}.json"
            if not p.exists(): continue
            lats=_lats(_load(str(p)),mt)
            if lats: xs.append(LOAD_RATES[level]); ys.append(_pct(lats,.99))
        if xs: ax.plot(xs,ys,color=COLORS[mode],marker="o",linewidth=2,
                       linestyle="--" if mode in("fifo","static_batch") else "-",label=LABELS[mode])
    ax.axhline(SLA[mt],color="#E24B4A",linewidth=1.2,linestyle=":",label=f"SLA {SLA[mt]}ms")
    ax.set_xlabel("Arrival rate (req/s)"); ax.set_ylabel("P99 latency (ms)")
    ax.set_title(f"P99 vs load — {mt.upper()}"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    out=f"{out_dir}/fig2_p99_load_{mt}.png"; fig.savefig(out,bbox_inches="tight"); plt.close()
    print(f"  {out}")

# Figure 3: SLA violation rate vs load (MAIN RESULT)
def fig_sla_violation(log_dir, out_dir="results/plots"):
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,mt in zip(axes,["icu","nlp","imaging"]):
        for mode in ["fifo","static_batch","triton","medserve"]:
            xs,ys=[],[]
            for level in LOAD_LEVELS:
                p=Path(log_dir)/f"{mode}_{level}.json"
                if not p.exists(): continue
                xs.append(LOAD_RATES[level]); ys.append(_viol(_load(str(p)),mt))
            if xs: ax.plot(xs,ys,color=COLORS[mode],marker="o",linewidth=2,
                           linestyle="--" if mode in("fifo","static_batch") else "-",label=LABELS[mode])
        ax.axhline(5,color="gray",linewidth=.8,linestyle=":",alpha=.7)
        ax.set_title(f"{mt.upper()} (SLA={SLA[mt]}ms)"); ax.set_xlabel("req/s")
        ax.set_ylim(-1,101); ax.grid(alpha=.3)
        if mt=="icu": ax.set_ylabel("SLA violation %")
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.08),fontsize=8)
    fig.suptitle("SLA violation rate vs load (lower = better)"); fig.tight_layout()
    out=f"{out_dir}/fig3_sla_violation.png"; fig.savefig(out,bbox_inches="tight"); plt.close()
    print(f"  {out}")

# Figure 4: Throughput vs P99 scatter
def fig_tradeoff(log_dir, out_dir="results/plots"):
    fig,ax=plt.subplots(figsize=(7,5))
    markers={"low":"o","medium":"s","high":"^"}
    for mode in ["fifo","static_batch","triton","medserve"]:
        for level in LOAD_LEVELS:
            p=Path(log_dir)/f"{mode}_{level}.json"
            if not p.exists(): continue
            recs=_load(str(p)); lats=_lats(recs)
            if not lats: continue
            p99=_pct(lats,.99)
            span=max((recs[-1]["arrival_offset_ms"]-recs[0]["arrival_offset_ms"])/1000,1)
            rps=len(recs)/span
            ax.scatter(rps,p99,color=COLORS[mode],s=90,marker=markers[level],zorder=3)
        ax.plot([],[],color=COLORS[mode],linewidth=2,label=LABELS[mode])
    for mk,lbl in [("o","Low"),("s","Medium"),("^","High")]:
        ax.scatter([],[],c="gray",marker=mk,s=60,label=f"{lbl} load")
    ax.set_xlabel("Throughput (req/s)"); ax.set_ylabel("P99 latency (ms)")
    ax.set_title("Throughput vs P99 tradeoff"); ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.3)
    fig.tight_layout()
    out=f"{out_dir}/fig4_tradeoff.png"; fig.savefig(out,bbox_inches="tight"); plt.close()
    print(f"  {out}")

# Figure 5: Ablation bars
def fig_ablation(ablation_map: Dict[str,str], level="medium", out_dir="results/plots"):
    mtypes=["icu","nlp","imaging"]
    names=list(ablation_map.keys())
    x=np.arange(len(names)); w=0.25
    clrs={"icu":"#E24B4A","nlp":"#378ADD","imaging":"#1D9E75"}
    fig,ax=plt.subplots(figsize=(10,5))
    for i,mt in enumerate(mtypes):
        viols=[]
        for path in ablation_map.values():
            viols.append(_viol(_load(path),mt) if Path(path).exists() else 0)
        bars=ax.bar(x+i*w,viols,w,label=f"{mt.upper()} SLA={SLA[mt]}ms",
                    color=clrs[mt],alpha=.85)
        for b,v in zip(bars,viols):
            if v>0: ax.text(b.get_x()+b.get_width()/2,b.get_height()+.5,
                             f"{v:.1f}%",ha="center",va="bottom",fontsize=8)
    ax.set_xticks(x+w); ax.set_xticklabels(names,rotation=15,ha="right",fontsize=9)
    ax.set_ylabel("SLA violation %"); ax.set_title(f"Ablation study — {level} load")
    ax.axhline(5,color="gray",linestyle=":",linewidth=.8,alpha=.6)
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=.3); fig.tight_layout()
    out=f"{out_dir}/fig5_ablation_{level}.png"; fig.savefig(out,bbox_inches="tight"); plt.close()
    print(f"  {out}")

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--log-dir",default="results/logs")
    p.add_argument("--out-dir",default="results/plots")
    args=p.parse_args()
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    print(f"\nGenerating figures from {args.log_dir} ...\n")
    for lv in LOAD_LEVELS: fig_cdf(args.log_dir,lv,args.out_dir)
    for mt in ["icu","nlp","imaging"]: fig_p99_load(args.log_dir,mt,args.out_dir)
    fig_sla_violation(args.log_dir,args.out_dir)
    fig_tradeoff(args.log_dir,args.out_dir)
    abl={
        "Full MedServe":f"{args.log_dir}/medserve_medium.json",
        "FIFO":         f"{args.log_dir}/fifo_medium.json",
        "Static batch": f"{args.log_dir}/static_batch_medium.json",
        "Triton":       f"{args.log_dir}/triton_medium.json",
    }
    if any(Path(v).exists() for v in abl.values()):
        fig_ablation(abl,"medium",args.out_dir)
    print(f"\nDone. Figures in: {args.out_dir}/")
