"""Audit and summarize matched online Thompson/Q comparisons in every case."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "robustness"))
sys.path.insert(0, str(HERE.parent / "feedback"))
from run_suite import load_gzip, make_rows
from analyze import bootstrap_mean, write_csv

MODELS = ("thompson", "reward", "both")
METRICS = ("app_delivery", "pdr", "j_per_delivery", "dl_airtime_s",
           "dl_airtime_per1000_generated_s", "total_j", "reports", "dl_sent",
           "resolved", "unresolved", "tail_unknown", "mean_label_delay_s")
LABELS = {"thompson": "Delayed binary Thompson", "reward": "Energy-reward Q",
          "both": "Delivery-state + energy Q"}
COLORS = {"thompson": "#8c6bb1", "reward": "#377eb8", "both": "#15936d"}


def verify(index, records, training):
    config = index["config"]
    expected = {(c, s, l, a) for c in config["cases"] for s in config["seeds"]
                for l in MODELS for a in ("A", "D")}
    actual = [(r["scenario"], r["seed"], r["learner"], r["condition"]) for r in records]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("Incomplete or duplicated online baseline matrix")
    expected_train = {(s, l, a, ep) for s in config["seeds"] for l in MODELS
                      for a in ("A", "D") for ep in range(config["train_episodes"])}
    actual_train = [(r["seed"], r["learner"], r["condition"], r["episode"]) for r in training]
    if set(actual_train) != expected_train or len(actual_train) != len(set(actual_train)):
        raise ValueError("Incomplete or duplicated baseline training matrix")
    if set(config["seeds"]) & {r["environment_seed"] for r in training}:
        raise ValueError("Training/evaluation seed overlap")
    traces, initial = {}, {}
    for r in records:
        key = r["scenario"], r["seed"]
        value = r["trace_sha256"], r["topology_sha256"]
        if traces.setdefault(key, value) != value:
            raise ValueError("Compared models have different exogenous environments")
        key = r["seed"], r["learner"], r["condition"]
        if initial.setdefault(key, r["initial_policy_sha256"]) != r["initial_policy_sha256"]:
            raise ValueError("Cases did not reuse the same pretrained prior")
        if r["nodes"] != config["nodes"] or r["mode"] != "online":
            raise ValueError("Unmatched evaluation mode or density")
        resolved = 0
        for p in r["phases"]:
            if not 0 <= p["received"] <= p["attempted"] <= p["generated"]:
                raise ValueError("Packet accounting mismatch")
            if p["resolved"] + p["unresolved"] + p["tail_unknown"] != p["attempted"]:
                raise ValueError("Packet-label accounting mismatch")
            if not np.isclose(p["total_j"], p["tx_j"]+p["rx_j"]+p["sleep_j"], atol=1e-12):
                raise ValueError("Energy partition mismatch")
            resolved += p["resolved"]
        if r["evaluation_updates"] != resolved:
            raise ValueError("Updates do not match received labels")
        if r["scenario"] == "no_feedback":
            if resolved or r["evaluation_updates"] or sum(p["reports"] for p in r["phases"]):
                raise ValueError("Missing reports supplied learning outcomes")
            if r["learner"] == "thompson" and r["initial_policy_sha256"] != r["final_policy_sha256"]:
                raise ValueError("Thompson posterior changed without observations")
    return dict(status="passed", evaluations=len(records), training_episodes=len(training),
                paired_seeds=len(config["seeds"]), environment_groups=len(traces),
                initial_prior_groups=len(initial), bootstrap_resamples=10000,
                statistical_unit="evaluation seed", comparisons="pointwise paired percentile bootstrap")


def summarize(rows):
    families = defaultdict(dict)
    for r in rows:
        families[r["scenario"], r["phase"]][r["seed"], r["learner"], r["condition"]] = r
    summary, paired = [], []
    for key, selected in sorted(families.items(), key=lambda item: str(item[0])):
        seeds = sorted({s for s, l, a in selected})
        for learner in MODELS:
            for arm in ("A", "D"):
                for metric in METRICS:
                    values = [selected[s, learner, arm].get(metric) for s in seeds]
                    available = [x for x in values if x is not None]
                    if metric in ("app_delivery", "pdr", "j_per_delivery") and len(available) != len(seeds):
                        raise ValueError("Undefined primary denominator")
                    mean, lo, hi = bootstrap_mean(available, f"online_summary:{key}:{learner}:{arm}:{metric}") if available else (None,)*3
                    summary.append(dict(scenario=key[0], phase=key[1], learner=learner,
                        condition=arm, metric=metric, mean=mean, ci95_low=lo, ci95_high=hi,
                        n_total=len(seeds), n_valid=len(available)))
        contrasts = [("D-A", learner, None) for learner in MODELS]
        contrasts += [(f"{q}-thompson_{arm}", q, arm) for q in ("reward", "both") for arm in ("A", "D")]
        for contrast, learner, arm in contrasts:
            for metric in METRICS:
                values = []
                for seed in seeds:
                    right = selected[seed, learner, "A"] if arm is None else selected[seed, "thompson", arm]
                    left = selected[seed, learner, "D"] if arm is None else selected[seed, learner, arm]
                    if left.get(metric) is None or right.get(metric) is None:
                        continue
                    values.append(left[metric]-right[metric])
                if metric in ("app_delivery", "pdr", "j_per_delivery") and len(values) != len(seeds):
                    raise ValueError("Undefined paired primary denominator")
                mean, lo, hi = bootstrap_mean(values, f"online_paired:{key}:{learner}:{contrast}:{metric}") if values else (None,)*3
                paired.append(dict(scenario=key[0], phase=key[1], learner=learner,
                    contrast=contrast, metric=metric, delta_mean=mean, ci95_low=lo,
                    ci95_high=hi, n_total=len(seeds), n_valid=len(values)))
    return summary, paired


def figures(summary, paired, config, output):
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    lookup = {(r["scenario"], r["learner"], r["condition"], str(r["phase"]), r["metric"]): r for r in summary}
    pairs = {(r["scenario"], r["learner"], str(r["phase"]), r["contrast"], r["metric"]): r for r in paired}
    cases = config["cases"]

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(output / f"{name}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    for ax, scenario in zip(axes.flat, cases):
        for learner in MODELS:
            for arm, marker in (("A", "o"), ("D", "s")):
                x = lookup[scenario, learner, arm, "total", "j_per_delivery"]
                y = lookup[scenario, learner, arm, "total", "app_delivery"]
                xv, yv = 1000*x["mean"], 100*y["mean"]
                ax.errorbar(xv, yv, xerr=[[max(0, xv-1000*x["ci95_low"])], [max(0, 1000*x["ci95_high"]-xv)]],
                    yerr=[[max(0, yv-100*y["ci95_low"])], [max(0, 100*y["ci95_high"]-yv)]],
                    fmt=marker, color=COLORS[learner], markerfacecolor="white" if arm == "A" else COLORS[learner],
                    capsize=2, markersize=6)
        ax.set(title=scenario.replace("_", " "), xlabel="Energy / delivered packet (mJ)", ylabel="Application delivery (%)", ylim=(0, 100))
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(cases):]:
        ax.set_axis_off()
    handles = [Line2D([0],[0],color=COLORS[m],marker="o",ls="",label=LABELS[m]) for m in MODELS]
    handles += [Line2D([0],[0],color="black",marker="o",markerfacecolor="white",ls="",label="A: rich + request"),
                Line2D([0],[0],color="black",marker="s",ls="",label="D: compact, no request")]
    fig.suptitle(f"Matched online baselines: delivery–energy trade-off at {config['nodes']} devices", fontsize=15)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, .95))
    save(fig, "01_online_delivery_energy")

    metrics = (("app_delivery",100.,"Delivery change (percentage points)"),
               ("j_per_delivery",1000.,"Energy change (mJ / delivery)"),
               ("dl_airtime_per1000_generated_s",1.,"DL change (s / 1,000 generated)"))
    fig, axes = plt.subplots(len(cases),3,figsize=(14,max(5,2.6*len(cases))),squeeze=False)
    for row, scenario in enumerate(cases):
        for col, (metric, factor, title) in enumerate(metrics):
            ax=axes[row,col]
            for i, learner in enumerate(MODELS):
                r=pairs[scenario,learner,"total","D-A",metric]
                v,lo,hi=[r[k]*factor for k in ("delta_mean","ci95_low","ci95_high")]
                ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt="o",color=COLORS[learner],capsize=3)
            ax.axvline(0,color="#777777",ls=":")
            ax.set_yticks(range(3),["Thompson","Reward Q","Combined Q"])
            ax.set(title=scenario.replace("_"," "),xlabel=title)
            ax.grid(alpha=.2)
    fig.suptitle("Effect of compact request-free feedback: D minus A, all online models",fontsize=14,y=.998)
    fig.tight_layout(rect=(0,.025,1,.985))
    fig.text(.5,.008,"Pointwise 95% paired seed-bootstrap intervals. Positive delivery and negative energy/airtime favor D; all cases retained.",ha="center",fontsize=9)
    save(fig,"02_online_feedback_effects")

    fig,axes=plt.subplots(2,2,figsize=(14,11))
    labels=[f"{c.replace('_',' ')} / {a}" for c in cases for a in ("A","D")]
    for row,learner in enumerate(("reward","both")):
        for col,(metric,factor,title) in enumerate(metrics[:2]):
            ax=axes[row,col]
            for i,(scenario,arm) in enumerate((c,a) for c in cases for a in ("A","D")):
                r=pairs[scenario,learner,"total",f"{learner}-thompson_{arm}",metric]
                v,lo,hi=[r[k]*factor for k in ("delta_mean","ci95_low","ci95_high")]
                ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt="o" if arm=="A" else "s",color=COLORS[learner],capsize=3)
            ax.axvline(0,color="#777777",ls=":")
            ax.set_yticks(range(len(labels)),labels,fontsize=8)
            ax.invert_yaxis()
            ax.set(title=f"{LABELS[learner]} minus Thompson",xlabel=title)
            ax.grid(alpha=.2)
    fig.suptitle("Matched-protocol comparison to the delayed binary Thompson reference",fontsize=14,y=.995)
    fig.tight_layout(rect=(0,.025,1,.98))
    fig.text(.5,.008,"Objectives differ: Thompson maximizes observed reception probability; Q reward includes energy. These are trade-offs, not universal rankings.",ha="center",fontsize=9)
    save(fig,"03_q_vs_thompson")


def report(config, summary, paired, checks, path):
    lookup={(r["scenario"],r["learner"],r["condition"],str(r["phase"]),r["metric"]):r for r in summary}
    lines=["# Matched online baseline results","",
        "Executed synthetic results: all three models learn locally during evaluation and use the same30 SF–power options and costed feedback protocols. Thompson uses Beta(1,1) priors and binary reception observations; the Q learners use delivery/energy reward. The Thompson reference is an adaptation, not an exact Navas energy-aware policy or paper replication.","",
        f"The complete matrix contains **{checks['evaluations']} evaluations**, **{checks['training_episodes']} training episodes**, and **{checks['paired_seeds']} paired evaluation seeds** at **{config['nodes']} devices**. All six declared cases are retained when using the default configuration.","",
        "## Compact request-free deployment, condition D","",
        "| Case | Online model | Application delivery | PDR/attempt | Energy (mJ/delivery) | DL airtime (s) |",
        "|---|---|---:|---:|---:|---:|"]
    for case in config["cases"]:
        for model in MODELS:
            values=[lookup[case,model,"D","total",metric]["mean"] for metric in ("app_delivery","pdr","j_per_delivery","dl_airtime_s")]
            lines.append(f"| {case} | {LABELS[model]} | {100*values[0]:.2f}% | {100*values[1]:.2f}% | {1000*values[2]:.2f} | {values[3]:.2f} |")
    lines += ["","## Paired feedback effects, D minus A","",
        "| Case | Online model | Delivery change (percentage points) | Energy change (mJ/delivery) |",
        "|---|---|---:|---:|"]
    pairs={(r["scenario"],r["learner"],str(r["phase"]),r["contrast"],r["metric"]):r for r in paired}
    for case in config["cases"]:
        for model in MODELS:
            cells=[]
            for metric,factor in (("app_delivery",100.),("j_per_delivery",1000.)):
                r=pairs[case,model,"total","D-A",metric]
                cells.append(f"{factor*r['delta_mean']:+.2f} [{factor*r['ci95_low']:+.2f}, {factor*r['ci95_high']:+.2f}]")
            lines.append(f"| {case} | {LABELS[model]} | "+" | ".join(cells)+" |")
    lines += ["","## Interpretation","",
        "- All intervals are pointwise95% paired seed-bootstrap intervals, without multiple-comparison adjustment. An interval crossing zero is inconclusive, not evidence of equivalence.",
        "- Initial tables/posteriors are reused across scenarios; each device receives an independent local copy. There is no free cross-device update sharing during deployment.",
        "- Training uses the normal load-step case separately for A/D. Therefore D−A includes training-protocol and deployment effects. Training radio energy is reported separately.",
        "- In the all-feedback-erased negative control, no learning outcome reaches any model. Sent reports still consume gateway airtime and receive energy.",
        "- Thompson and Q have different optimization objectives. Read delivery and energy together, including cases where the Q extension loses to the bandit. Do not select only favorable cases or infer universal superiority.",
        "- The reconstructed DSF2R and state-only ablation are evaluated in the separate frozen robustness study. No frozen-versus-online comparison here is labelled a causal algorithm improvement.",
        "- Receiver calibration, independent ns-3 validation and hardware measurements remain outside this synthetic study.","",
        "Every model/protocol/phase and paired Q−Thompson contrast is in the CSV files; compressed raw checkpoints and trained priors are retained.",""]
    path.write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=HERE/"results/index.json")
    parser.add_argument("--figures",type=Path,default=HERE/"figures")
    args=parser.parse_args()
    index=json.loads(args.input.read_text(encoding="utf-8"))
    for name,digest in index["source_sha256"].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Source changed since run: {name}")
    records,training=[],[]
    for relative in index["chunks"]:
        path=args.input.parent/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=index["chunk_sha256"][relative]:
            raise ValueError("Changed raw checkpoint")
        chunk=load_gzip(path)
        records.extend(chunk["records"]);training.extend(chunk["training"])
    checks=verify(index,records,training)
    rows=make_rows(records)
    summary,paired=summarize(rows)
    output=args.input.parent
    for name,data in (("per_seed.csv",rows),("training.csv",training),("summary.csv",summary),("paired_differences.csv",paired)):
        write_csv(output/name,data)
    (output/"analysis_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    report(index["config"],summary,paired,checks,output/"RESULTS.md")
    figures(summary,paired,index["config"],args.figures)
    print(json.dumps(checks,indent=2),flush=True)


if __name__=="__main__":
    main()
