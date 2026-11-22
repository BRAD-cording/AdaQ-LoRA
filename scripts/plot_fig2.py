#!/usr/bin/env python
"""Redraw Fig. 2 (epsilon profiles) with honest axes.

Fixes: (i) panels (a) and (b) share one y-range so the visual impression matches
the coefficient of variation; (ii) CV computed with the sample standard
deviation, matching the manuscript; (iii) the cross-quantizer correlations are
labelled with their own n, so the 32-point q_proj figure is not confused with
the 64-point scatter.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/ubuntu/AdaQ-LoRA"
FILES = {
    "NF4": "results_v2/epsilon_l_report.json",
    "INT4-RTN": "results_v2/epsilon_report_int4rtn_g128.json",
}
OUT = ROOT + "/paper/epsilon_cross_quantizer.png"


def load(fname):
    d = json.load(open(ROOT + "/" + fname))
    pl = d["per_layer"]
    q, v = {}, {}
    for k, e in pl.items():
        layer = int(k.split("layers.")[1].split(".")[0])
        (q if "q_proj" in k else v)[layer] = e["eps"]
    keys = sorted(q)
    return {
        "eps_q": [q[i] for i in keys],
        "eps_v": [v[i] for i in keys],
        "eps_all": [q[i] for i in keys] + [v[i] for i in keys],
    }


def cv(x):
    return 100.0 * float(np.std(x, ddof=1)) / float(np.mean(x))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


nf4, rtn = load(FILES["NF4"]), load(FILES["INT4-RTN"])
ylim = (min(min(nf4["eps_all"]), min(rtn["eps_all"])) - 0.0006,
        max(max(nf4["eps_all"]), max(rtn["eps_all"])) + 0.0006)

fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))
x = np.arange(32)

ax = axes[0][0]
ax.plot(x, nf4["eps_q"], "o-", color="#1f77b4", ms=4, label="q_proj")
ax.plot(x, nf4["eps_v"], "s--", color="#d62728", ms=4, label="v_proj")
ax.axhline(float(np.mean(nf4["eps_all"])), color="gray", ls=":", lw=1,
           label="mean=%.4f" % np.mean(nf4["eps_all"]))
ax.set_title("(a) NF4:  CV=%.2f%%, max/min=%.3f" % (
    cv(nf4["eps_all"]), max(nf4["eps_all"]) / min(nf4["eps_all"])))
ax.set_xlabel("layer index"); ax.set_ylabel(r"$\varepsilon_l$")
ax.set_ylim(*ylim); ax.legend(fontsize=8); ax.grid(alpha=.25)

ax = axes[0][1]
ax.plot(x, rtn["eps_q"], "o-", color="#1f77b4", ms=4, label="q_proj")
ax.plot(x, rtn["eps_v"], "s--", color="#d62728", ms=4, label="v_proj")
ax.axhline(float(np.mean(rtn["eps_all"])), color="gray", ls=":", lw=1,
           label="mean=%.4f" % np.mean(rtn["eps_all"]))
ax.set_title("(b) INT4-RTN g128:  CV=%.2f%%, max/min=%.3f" % (
    cv(rtn["eps_all"]), max(rtn["eps_all"]) / min(rtn["eps_all"])))
ax.set_xlabel("layer index"); ax.set_ylabel(r"$\varepsilon_l$")
ax.set_ylim(*ylim); ax.legend(fontsize=8); ax.grid(alpha=.25)

ax = axes[1][0]
rho_q = spearman(nf4["eps_q"], rtn["eps_q"])
ax.plot(x, nf4["eps_q"], "o-", color="#1f77b4", ms=4, label="NF4 q_proj")
ax.plot(x, rtn["eps_q"], "^--", color="#ff7f0e", ms=4, label="INT4-RTN q_proj")
ax.set_title(r"(c) q_proj only, $n=32$:  $\rho=%.3f$" % rho_q)
ax.set_xlabel("layer index"); ax.set_ylabel(r"$\varepsilon_l$")
ax.legend(fontsize=8); ax.grid(alpha=.25)

ax = axes[1][1]
rho_all = spearman(nf4["eps_all"], rtn["eps_all"])
ax.scatter(nf4["eps_all"], rtn["eps_all"], s=22, color="#1f77b4", alpha=.75)
lo = min(min(nf4["eps_all"]), min(rtn["eps_all"])) - 0.001
hi = max(max(nf4["eps_all"]), max(rtn["eps_all"])) + 0.001
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
ax.set_title(r"(d) all 64 modules, $n=64$:  $\rho=%.3f$" % rho_all)
ax.set_xlabel("NF4 " + r"\varepsilon_l$")
ax.set_ylabel("INT4-RTN " + r"\varepsilon_l$")
ax.legend(fontsize=8); ax.grid(alpha=.25)

fig.suptitle(r"Per-layer relative quantization error $\varepsilon_l$ on Llama-3-8B "
             r"(panels a,b share one y-range)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
print("CV_NF4=%.3f%% CV_INT4=%.3f%%" % (cv(nf4["eps_all"]), cv(rtn["eps_all"])))
print("maxmin_NF4=%.4f maxmin_INT4=%.4f" % (
    max(nf4["eps_all"]) / min(nf4["eps_all"]), max(rtn["eps_all"]) / min(rtn["eps_all"])))
print("mean_NF4=%.5f mean_INT4=%.5f" % (np.mean(nf4["eps_all"]), np.mean(rtn["eps_all"])))
print("rho_q32=%.4f rho_all64=%.4f" % (rho_q, rho_all))
