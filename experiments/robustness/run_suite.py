"""Frozen-policy stress study with paired environments and explicit feedback costs.

Run --quick for the mechanics check or run without arguments for the registered
six-case, three-density, thirty-seed matrix. Historical feedback code is imported
unchanged. This is a synthetic robustness study, not an ns-3 validation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import copy
from dataclasses import dataclass, asdict, replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FEEDBACK = HERE.parent / "feedback"
sys.path.insert(0, str(FEEDBACK))
import engine
from policies import QPolicy
from run_experiment import q_sha256, save_q
from analyze import bootstrap_mean, derived, write_csv

BASE_MAKE_TRACE = engine.make_trace
LEARNERS = ("original", "reward", "state", "both")
MODELS = LEARNERS + ("fixed_conservative", "fixed_no_feedback")
METRICS = ("app_delivery", "pdr", "j_per_delivery", "dl_airtime_s",
           "dl_airtime_per1000_generated_s", "total_j", "reports", "dl_sent",
           "resolved", "unresolved", "tail_unknown", "mean_label_delay_s")
NAMES = {"original": "DSF2R reconstruction", "reward": "Energy reward only",
         "state": "Delivery state only", "both": "Delivery state + energy reward",
         "fixed_conservative": "Fixed SF12 / 14 dBm, matched A",
         "fixed_no_feedback": "Fixed SF12 / 14 dBm, no feedback"}


@dataclass(frozen=True)
class CaseSpec:
    intervals: tuple[float, float, float]
    downlink_loss: float = .05
    attenuation_db: float = 0.
    attenuation_start_phase: int = 1
    description: str = ""


CASES = {
    "stable_low": CaseSpec((400., 400., 400.), description="Light periodic traffic in all phases"),
    "stable_high": CaseSpec((160., 160., 160.), description="Heavy periodic traffic in all phases"),
    "load_step": CaseSpec((400., 160., 400.), description="Light to heavy to light traffic"),
    "channel_step": CaseSpec((400., 160., 400.), attenuation_db=6.,
        description="Load step plus 6 dB UL/DL attenuation from phase 1 onward"),
    "lossy_feedback": CaseSpec((400., 160., 400.), downlink_loss=.30,
        description="Load step with 30% additional downlink erasure"),
    "no_feedback": CaseSpec((400., 160., 400.), downlink_loss=1.,
        description="All feedback erased; outcomes remain unknown"),
}
SCENARIOS = CASES


class FixedConservative(QPolicy):
    def __init__(self):
        super().__init__(epsilon=0., update_enabled=False)

    def choose(self, state, rng):
        return 29  # SF12,14 dBm in the common thirty-option action set.


def make_case_trace(seed, n, phase_seconds, case_name):
    spec = CASES[case_name]
    if spec.intervals == (400., 160., 400.):
        positions, loss, events = BASE_MAKE_TRACE(seed, n, phase_seconds)
    else:
        rng = np.random.default_rng(seed)
        radii = np.sqrt(rng.uniform(200**2, 2600**2, n))
        angles = rng.uniform(0, 2*np.pi, n)
        positions = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))
        loss = 32.4 + 32*np.log10(radii) + rng.normal(0, 3, n)
        events = []
        for phase, interval in enumerate(spec.intervals):
            for device in range(n):
                offset = rng.uniform(0, interval)
                for t in np.arange(phase*phase_seconds+offset, (phase+1)*phase_seconds, interval):
                    events.append((float(t), device, int(rng.integers(3)),
                                   float(rng.normal(0, 1.5)), float(rng.random()),
                                   float(rng.normal(0, 1.5))))
        events.sort()
    if spec.attenuation_db:
        threshold = phase_seconds*spec.attenuation_start_phase
        events = [(t, d, c, f-(spec.attenuation_db if t >= threshold else 0.), u,
                   df-(spec.attenuation_db if t >= threshold else 0.))
                  for t, d, c, f, u, df in events]
    return positions, loss, events


@contextmanager
def case_trace(case_name):
    if case_name not in CASES:
        raise ValueError(f"Unknown case: {case_name}")
    previous = engine.make_trace
    engine.make_trace = lambda seed, n, seconds: make_case_trace(seed, n, seconds, case_name)
    try:
        yield
    finally:
        engine.make_trace = previous


def run_case(seed, n, phase_seconds, policy, variant, case_name, condition="A", training=False, debug=False):
    if condition == "N":
        feedback = engine.FeedbackConfig("rich", 0, interval=10**9,
                                         downlink_loss=CASES[case_name].downlink_loss)
    else:
        feedback = replace(engine.CONFIGS[condition], downlink_loss=CASES[case_name].downlink_loss)
    with case_trace(case_name):
        return engine.run(seed, n, phase_seconds, policy, variant, feedback=feedback,
                          training=training, debug=debug)


def sources():
    names = ["experiments/feedback/engine.py", "experiments/feedback/policies.py",
             "experiments/feedback/feedback_codec.py", "experiments/feedback/run_experiment.py",
             "experiments/feedback/analyze.py", "experiments/robustness/run_suite.py"]
    return {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in names}


def dump_gzip(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def load_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def train_common(seed, nodes, args, output):
    policies, training = {}, []
    for learner in LEARNERS:
        policy = QPolicy(alpha=.1, gamma=.9, epsilon=.25)
        for episode in range(args.train_episodes):
            environment_seed = seed+50000+1000*episode
            before = policy.update_count
            run = run_case(environment_seed, nodes, args.phase_seconds, policy, learner,
                           "load_step", "A", training=True)
            training.append(dict(seed=seed, nodes=nodes, learner=learner, episode=episode,
                environment_seed=environment_seed, scenario="load_step", condition="A",
                total_j=sum(p["total_j"] for p in run["phases"]),
                attempted=sum(p["attempted"] for p in run["phases"]),
                received=sum(p["received"] for p in run["phases"]),
                updates=policy.update_count-before, trace_sha256=run["trace_sha256"],
                topology_sha256=run["topology_sha256"], table_sha256=q_sha256(policy)))
        policies[learner] = policy
        save_q(policy, output/"tables"/f"n{nodes}_s{seed}_{learner}.npz")
    return policies, training


def evaluate_group(seed, nodes, args, output):
    policies, training = train_common(seed, nodes, args, output)
    records, topologies = [], {}
    for scenario in args.cases:
        expected_trace = None
        for learner in MODELS:
            conditions = (("N",) if learner == "fixed_no_feedback" else
                          ("A",) if learner == "fixed_conservative" else tuple(engine.CONFIGS))
            for condition in conditions:
                policy = FixedConservative() if learner.startswith("fixed_") else copy.deepcopy(policies[learner])
                policy.epsilon, policy.update_enabled = 0., False
                initial_hash, before = q_sha256(policy), policy.update_count
                variant = "original" if learner.startswith("fixed_") else learner
                record = run_case(seed, nodes, args.phase_seconds, policy, variant, scenario, condition)
                final_hash = q_sha256(policy)
                if initial_hash != final_hash or record["updates"] != before:
                    raise AssertionError("Frozen policy changed")
                signature = (record["trace_sha256"], record["topology_sha256"])
                if expected_trace is not None and signature != expected_trace:
                    raise AssertionError("Unpaired within-case environment")
                expected_trace = signature
                topologies.setdefault(record["topology_sha256"], record.pop("positions"))
                record.update(mode="frozen", scenario=scenario, seed=seed, nodes=nodes,
                    learner=learner, condition=condition, initial_q_sha256=initial_hash,
                    final_q_sha256=final_hash, evaluation_updates=record["updates"]-before)
                for p in record["phases"]:
                    if p["resolved"]+p["unresolved"]+p["tail_unknown"] != p["attempted"]:
                        raise AssertionError("Packet-label accounting mismatch")
                    if scenario == "no_feedback" and (p["reports"] or p["resolved"]):
                        raise AssertionError("Erased feedback supplied labels")
                records.append(record)
    return dict(nodes=nodes, seed=seed, training=training, records=records, topologies=topologies)


def make_rows(records):
    rows = []
    means = {"pdr", "app_delivery", "j_per_delivery", "feedback_age_mean_s"}
    for r in records:
        meta = {k: r[k] for k in ("scenario", "nodes", "seed", "learner", "condition", "mode")}
        total = defaultdict(float)
        for phase, stats in enumerate(r["phases"]):
            rows.append({**meta, "phase": phase, **derived(stats)})
            for key, value in stats.items():
                if key not in means and isinstance(value, (int, float)):
                    total[key] += value
        rows.append({**meta, "phase": "total", **derived(dict(total))})
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ("scenario", "nodes", "learner", "condition", "phase"))].append(r)
    summary, paired = [], []
    for key, selected in sorted(groups.items(), key=lambda item: str(item[0])):
        selected.sort(key=lambda r: r["seed"])
        for metric in METRICS:
            values = [r.get(metric) for r in selected if r.get(metric) is not None]
            avg, lo, hi = bootstrap_mean(values, f"stress_summary:{key}:{metric}") if values else (None, None, None)
            summary.append({**dict(zip(("scenario", "nodes", "learner", "condition", "phase"), key)),
                "metric": metric, "n_total": len(selected), "n_valid": len(values),
                "mean": avg, "ci95_low": lo, "ci95_high": hi})
    by_family = defaultdict(dict)
    for r in rows:
        by_family[r["scenario"], r["nodes"], r["learner"], r["phase"]][r["seed"], r["condition"]] = r
    contrasts = {"B-A": {"B": 1, "A": -1}, "C-A": {"C": 1, "A": -1},
                 "D-A": {"D": 1, "A": -1}, "D-B": {"D": 1, "B": -1},
                 "D-C": {"D": 1, "C": -1}, "interaction_D-C-B+A": {"D": 1, "C": -1, "B": -1, "A": 1}}
    for key, selected in sorted(by_family.items(), key=lambda item: str(item[0])):
        if key[2].startswith("fixed_"):
            continue
        seeds = sorted({s for s, c in selected})
        for contrast, weights in contrasts.items():
            for metric in METRICS:
                values = []
                for seed in seeds:
                    if any(selected[seed, c].get(metric) is None for c in weights):
                        continue
                    values.append(sum(w*selected[seed, c][metric] for c, w in weights.items()))
                avg, lo, hi = bootstrap_mean(values, f"stress_paired:{key}:{contrast}:{metric}") if values else (None, None, None)
                paired.append({**dict(zip(("scenario", "nodes", "learner", "phase"), key)),
                    "contrast": contrast, "metric": metric, "n_total": len(seeds), "n_valid": len(values),
                    "delta_mean": avg, "ci95_low": lo, "ci95_high": hi})
    return summary, paired


def figures(summary, paired, config, output):
    os.environ.setdefault("MPLCONFIGDIR", str(HERE/".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    cases, nodes = config["cases"], config["node_counts"]
    sl = {(r["scenario"],r["nodes"],r["learner"],r["condition"],str(r["phase"]),r["metric"]):r for r in summary}
    pl = {(r["scenario"],r["nodes"],r["learner"],str(r["phase"]),r["contrast"],r["metric"]):r for r in paired}
    definitions = (("app_delivery",100.,"Delivery change (percentage points)",False),
                   ("j_per_delivery",1000.,"Energy change (mJ / delivery)",True),
                   ("dl_airtime_per1000_generated_s",1.,"DL change (s / 1,000 generated)",True))

    def save(fig,name):
        for ext in ("png","svg"):
            fig.savefig(output/f"{name}.{ext}",dpi=170,bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(4,3,figsize=(14,17))
    for row, learner in enumerate(LEARNERS):
        for col,(metric,factor,label,lower_good) in enumerate(definitions):
            ax=axes[row,col]
            values=np.array([[pl[c,n,learner,"total","D-A",metric]["delta_mean"]*factor for n in nodes] for c in cases])
            limit=max(float(np.nanmax(np.abs(values))),1e-9)
            ax.imshow(values,cmap="RdYlGn_r" if lower_good else "RdYlGn",vmin=-limit,vmax=limit,aspect="auto")
            ax.set_xticks(range(len(nodes)),nodes)
            ax.set_yticks(range(len(cases)),[c.replace('_',' ') for c in cases])
            ax.set(title=NAMES[learner]+"\n"+label,xlabel="Devices")
            for i,c in enumerate(cases):
                for j,n in enumerate(nodes):
                    r=pl[c,n,learner,"total","D-A",metric]
                    star="*" if r["ci95_low"]>0 or r["ci95_high"]<0 else ""
                    ax.text(j,i,f"{values[i,j]:+.2f}{star}",ha="center",va="center",fontsize=8,color="black")
    fig.suptitle("All stress cases: compact/no-request D minus rich/request A",fontsize=15,y=.995)
    fig.tight_layout(rect=(0,.025,1,.98))
    fig.text(.5,.008,"Green favors D. * Pointwise paired 95% bootstrap interval excludes zero; no multiplicity correction. Synthetic frozen-policy study.",ha="center",fontsize=9)
    save(fig,"01_case_density_effects")

    main_n=60 if 60 in nodes else nodes[0]
    colors={"original":"#666666","reward":"#377eb8","state":"#984ea3","both":"#15936d","fixed_conservative":"#e59c20","fixed_no_feedback":"#e41a1c"}
    short={"original":"DSF2R ref","reward":"Reward","state":"State","both":"Combined","fixed_conservative":"Fixed A","fixed_no_feedback":"Fixed N"}
    fig,axes=plt.subplots(len(cases),2,figsize=(13,max(5,2.7*len(cases))),squeeze=False)
    for i,c in enumerate(cases):
        for j,(metric,factor,label,_) in enumerate(definitions[:2]):
            ax=axes[i,j]
            for k,learner in enumerate(MODELS):
                arm="N" if learner=="fixed_no_feedback" else "A"
                r=sl[c,main_n,learner,arm,"total",metric]
                if r["mean"] is None:
                    continue
                mean,lo,hi=[r[x]*factor for x in ("mean","ci95_low","ci95_high")]
                ax.errorbar(k,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt="o",color=colors[learner],capsize=3)
            ax.set_xticks(range(len(MODELS)),[short[l] for l in MODELS])
            ax.set(title=c.replace('_',' '),ylabel="Application delivery (%)" if metric=="app_delivery" else "Energy / delivered packet (mJ)")
            ax.grid(alpha=.2)
            if metric=="app_delivery":ax.set_ylim(0,100)
    fig.suptitle(f"Allocation references and fixed controls at {main_n} devices",fontsize=14,y=.998)
    fig.tight_layout(rect=(0,.025,1,.985))
    fig.text(.5,.008,"Models use protocol A; Fixed N sends no learning feedback. Both fixed controls use SF12/14 dBm. Means and 95% seed-bootstrap intervals.",ha="center",fontsize=9)
    save(fig,"02_model_comparison")

    fig,axes=plt.subplots(2,3,figsize=(15,9),squeeze=False)
    for ax,c in zip(axes.flat,cases):
        for learner in MODELS:
            arms=("N",) if learner=="fixed_no_feedback" else (("A",) if learner=="fixed_conservative" else ("A","D"))
            for arm in arms:
                x=sl[c,main_n,learner,arm,"total","j_per_delivery"]
                y=sl[c,main_n,learner,arm,"total","app_delivery"]
                if x["mean"] is None or y["mean"] is None:continue
                marker="o" if arm=="A" else ("s" if arm=="D" else "^")
                ax.scatter(x["mean"]*1000,y["mean"]*100,color=colors[learner],marker=marker,s=45,facecolors="none" if arm=="A" else colors[learner])
        ax.set(title=c.replace('_',' '),xlabel="Energy / delivered packet (mJ)",ylabel="Application delivery (%)",ylim=(0,100))
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(cases):]:ax.set_axis_off()
    handles=[Line2D([0],[0],marker="o",ls="",color=colors[l],label=short[l]) for l in MODELS]
    handles += [Line2D([0],[0],marker="o",ls="",color="black",markerfacecolor="none",label="A: rich + request"),Line2D([0],[0],marker="s",ls="",color="black",label="D: compact, no request"),Line2D([0],[0],marker="^",ls="",color="black",label="N: no learning feedback")]
    fig.suptitle(f"Delivery–energy trade-offs at {main_n} devices (all stress cases)",fontsize=15)
    fig.legend(handles=handles,loc="lower center",ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.10,1,.95))
    save(fig,"03_delivery_energy_tradeoffs")

    contrasts=("B-A","C-A","D-A")
    fig,axes=plt.subplots(3,3,figsize=(14,15))
    labels=[f"{c.replace('_',' ')} / {n}" for c in cases for n in nodes]
    for row,(metric,factor,label,lower_good) in enumerate(definitions):
        matrices=[]
        for contrast in contrasts:
            matrices.append(np.array([pl[c,n,"both","total",contrast,metric]["delta_mean"]*factor for c in cases for n in nodes]).reshape(-1,1))
        limit=max(max(float(np.max(np.abs(x))) for x in matrices),1e-9)
        for col,(contrast,values) in enumerate(zip(contrasts,matrices)):
            ax=axes[row,col]
            ax.imshow(values,cmap="RdYlGn_r" if lower_good else "RdYlGn",vmin=-limit,vmax=limit,aspect="auto")
            ax.set_yticks(range(len(labels)),labels,fontsize=7)
            ax.set_xticks([])
            ax.set(title=contrast+"\n"+label)
            for i,(c,n) in enumerate((c,n) for c in cases for n in nodes):
                r=pl[c,n,"both","total",contrast,metric]
                star="*" if r["ci95_low"]>0 or r["ci95_high"]<0 else ""
                ax.text(0,i,f"{values[i,0]:+.2f}{star}",ha="center",va="center",fontsize=8)
    fig.suptitle("Combined learner: isolate compression, requests and their combination",fontsize=15,y=.995)
    fig.tight_layout(rect=(0,.025,1,.98))
    fig.text(.5,.008,"B−A: compression; C−A: request removal; D−A: both. Green favors the change; * pointwise paired interval excludes zero.",ha="center",fontsize=9)
    save(fig,"04_factor_effects")


def analyze(index_path, figure_path):
    index=json.loads(index_path.read_text(encoding="utf-8"))
    if index["source_sha256"] != sources():raise ValueError("Sources changed since run")
    records,training=[],[]
    for relative in index["chunks"]:
        path=index_path.parent/relative
        expected=index["chunk_sha256"][relative]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError("Changed raw chunk")
        chunk=load_gzip(path)
        records.extend(chunk["records"]);training.extend(chunk["training"])
    config=index["config"]
    expected={(c,n,s,l,a) for c in config["cases"] for n in config["node_counts"] for s in config["seeds"] for l in MODELS for a in (("N",) if l=="fixed_no_feedback" else (("A",) if l=="fixed_conservative" else tuple(engine.CONFIGS)))}
    actual={(r["scenario"],r["nodes"],r["seed"],r["learner"],r["condition"]) for r in records}
    if expected!=actual or len(actual)!=len(records):raise ValueError("Incomplete or duplicate evaluation matrix")
    signatures=defaultdict(set)
    tables=defaultdict(set)
    for r in records:
        signatures[r["scenario"],r["nodes"],r["seed"]].add((r["trace_sha256"],r["topology_sha256"]))
        tables[r["nodes"],r["seed"],r["learner"]].add(r["initial_q_sha256"])
        if r["initial_q_sha256"]!=r["final_q_sha256"] or r["evaluation_updates"]:raise ValueError("Frozen table mutation")
    if any(len(v)!=1 for v in signatures.values()) or any(len(v)!=1 for v in tables.values()):raise ValueError("Unpaired environments or tables")
    rows=make_rows(records)
    summary,paired=summarize(rows)
    output=index_path.parent
    for name,data in (("per_seed.csv",rows),("training.csv",training),("summary.csv",summary),("paired_differences.csv",paired)):
        write_csv(output/name,data)
    checks=dict(status="passed",evaluations=len(records),training_episodes=len(training),
        environment_groups=len(signatures),frozen_table_groups=len(tables),seeds_per_comparison=len(config["seeds"]),
        bootstrap_resamples=10000,statistical_unit="independent paired evaluation seed",
        primary_matrix_complete=True,source_hashes_verified=len(index["source_sha256"]))
    (output/"validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    lines=["# Frozen-policy robustness results","",
        f"Executed {len(records):,} evaluations and {len(training):,} training episodes across {len(config['seeds'])} paired seeds per comparison. All planned cases and feedback conditions are included.","",
        "This is the declared Python receiver/energy model, not ns-3 or hardware evidence. Policies are pretrained once on the load-step case and frozen across stress cases; settings still respond to observed states. Both fixed controls use SF12/14 dBm: one pays matched protocol A, the other sends no learning feedback (N). No faithful ADR implementation is claimed.","",
        "## Combined learner, D minus A over all phases","",
        "Negative energy/downlink differences favor D. Positive delivery differences favor D. Intervals spanning zero do not establish improvement or equivalence.","",
        "| Case | Devices | Delivery change (pp) | Energy change (mJ/delivery) | DL change (s/1,000 generated) |",
        "|---|---:|---:|---:|---:|"]
    lookup={(r["scenario"],r["nodes"],r["learner"],str(r["phase"]),r["contrast"],r["metric"]):r for r in paired}
    for c in config["cases"]:
        for n in config["node_counts"]:
            cells=[]
            for metric,factor in (("app_delivery",100),("j_per_delivery",1000),("dl_airtime_per1000_generated_s",1)):
                r=lookup[c,n,"both","total","D-A",metric]
                cells.append(f"{factor*r['delta_mean']:+.2f} [{factor*r['ci95_low']:+.2f}, {factor*r['ci95_high']:+.2f}]")
            lines.append(f"| {c} | {n} | "+" | ".join(cells)+" |")
    lines += ["","## Interpretation boundaries","",
        "- The all-feedback-erased case is a negative control: no receipts or SNR reach the device, though sent reports still consume airtime and receive energy.",
        "- Channel attenuation changes both directions from the second phase; the per-frame downlink fade is attached to the originating uplink, including any boundary crossing.",
        "- Frozen evaluations assess transfer of pretrained state-responsive policies, not online relearning after each change. Online baseline comparisons are a separate experiment.",
        "- Confidence intervals are pointwise paired seed bootstrap intervals, without multiplicity correction. Report all cases rather than select significant cells.",
        "- The table exposes successes, failures and inconclusive cases. It does not prove a universal research gap, optimality, or superiority to published implementations.",
        "- Exact intervals and every model/condition/phase are in summary.csv and paired_differences.csv; individual runs remain in per_seed.csv and compressed raw checkpoints.",""]
    (output/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    figures(summary,paired,config,figure_path)
    print(json.dumps(checks,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--nodes",type=int,nargs="+")
    parser.add_argument("--seeds",type=int,nargs="+")
    parser.add_argument("--phase-seconds",type=int)
    parser.add_argument("--train-episodes",type=int)
    parser.add_argument("--cases",nargs="+",choices=tuple(CASES),default=list(CASES))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--figures",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--analyze-only",action="store_true")
    args=parser.parse_args()
    args.nodes=args.nodes or ([8] if args.quick else [30,60,120])
    args.seeds=args.seeds or ([1001,1002] if args.quick else list(range(1001,1031)))
    args.phase_seconds=args.phase_seconds or (3200 if args.quick else 8000)
    args.train_episodes=args.train_episodes or (1 if args.quick else 3)
    output=args.output or HERE/("quick_results" if args.quick else "results")
    figure_path=args.figures or HERE/("quick_figures" if args.quick else "figures")
    if min(args.nodes)<1 or args.train_episodes<1 or args.phase_seconds<3200 or args.phase_seconds%800:
        parser.error("Positive counts; phase length>=3200 divisible by800")
    if len(set(args.nodes))!=len(args.nodes) or len(set(args.seeds))!=len(args.seeds) or len(set(args.cases))!=len(args.cases):
        parser.error("Do not repeat nodes, seeds or cases")
    train_seeds={s+50000+1000*ep for s in args.seeds for ep in range(args.train_episodes)}
    if train_seeds.intersection(args.seeds):parser.error("Training and evaluation seeds overlap")
    output.mkdir(parents=True,exist_ok=True)
    index_path=output/"index.json"
    if args.analyze_only:
        analyze(index_path,figure_path);return
    config=dict(node_counts=args.nodes,seeds=args.seeds,phase_seconds=args.phase_seconds,
        train_episodes=args.train_episodes,cases=args.cases,case_specs={c:asdict(CASES[c]) for c in args.cases},
        learners=list(MODELS),conditions={l:(["N"] if l=="fixed_no_feedback" else ["A"] if l=="fixed_conservative" else list(engine.CONFIGS)) for l in MODELS},
        common_training_case="load_step",common_training_condition="A",alpha=.1,gamma=.9,
        beta=engine.BETA,energy_reference_j=engine.E_REF,python_version=platform.python_version(),numpy_version=np.__version__,
        baseline_status="DSF2R reconstruction, state/reward ablations, and fixed conservative control; no exact ADR claim")
    config=json.loads(json.dumps(config))
    source=sources()
    if args.resume and index_path.exists():
        index=json.loads(index_path.read_text(encoding="utf-8"))
        if index["config"]!=config or index["source_sha256"]!=source:raise ValueError("Cannot resume changed configuration or sources")
    else:index=dict(config=config,source_sha256=source,chunks=[],chunk_sha256={})
    for nodes in args.nodes:
        for seed in args.seeds:
            relative=f"chunks/n{nodes}_s{seed}.json.gz"
            path=output/relative
            if args.resume and relative in index["chunks"]:
                if hashlib.sha256(path.read_bytes()).hexdigest()!=index["chunk_sha256"][relative]:raise ValueError("Changed checkpoint")
                print(f"Verified checkpoint N={nodes}, seed={seed}",flush=True)
                continue
            chunk=evaluate_group(seed,nodes,args,output)
            dump_gzip(path,chunk)
            if relative not in index["chunks"]:index["chunks"].append(relative)
            index["chunk_sha256"][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
            index_path.write_text(json.dumps(index,indent=2),encoding="utf-8")
            print(f"Saved six-case stress group N={nodes}, seed={seed}: {len(chunk['records'])} evaluations",flush=True)
    analyze(index_path,figure_path)


if __name__ == "__main__":
    main()
