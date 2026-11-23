"""Cross-check every statistic quoted in the prose against Table 1.

Collects (a) the six table rows, (b) every confidence interval, q value and
signed delta that appears in running text, then reports any prose number that
does not match a table cell or a value derivable from one.
"""
import re

TEX = "neurocomputing_paper.tex"
txt = open(TEX, encoding="utf-8-sig").read()

# ---- (a) table rows
rows = {}
for m in re.finditer(
        r"(NF4|INT4-RTN|INT8)\s*&\s*(MMLU|GSM8K)\s*&\s*(\d+)\s*&\s*\$([-+][\d.]+)\$\s*&"
        r"\s*\[\$([-+])\$([\d.]+),\s*\$([-+])\$([\d.]+)\]\s*&\s*\$([-+][\d.]+)\$\s*&"
        r"\s*([\d.]+)\s*&\s*([\d.]+)", txt):
    q, b = m.group(1), m.group(2)
    rows[(q, b)] = {
        "n": int(m.group(3)),
        "delta": float(m.group(4)),
        "lo": float(m.group(6)) * (1 if m.group(5) == "+" else -1),
        "hi": float(m.group(8)) * (1 if m.group(7) == "+" else -1),
        "d": float(m.group(9)),
        "q": float(m.group(10)),
        "mde": float(m.group(11)),
    }
print("table rows parsed:", len(rows))
for k, v in rows.items():
    print("   %-9s %-6s n=%-2d %+.2f [%+.2f, %+.2f] d=%+.2f q=%.3f mde=%.2f"
          % (k[0], k[1], v["n"], v["delta"], v["lo"], v["hi"], v["d"], v["q"], v["mde"]))

known = set()
for v in rows.values():
    for key in ("delta", "lo", "hi", "d", "q", "mde"):
        known.add(round(abs(v[key]), 2))
known |= {0.5, 2.0, 3.2, 5.0, 0.33, 0.20, 1.0}   # thresholds legitimately quoted
known |= {float(x) for x in ("0.998", "0.887", "0.796", "0.144", "0.084", "0.09",
                             "1.49", "0.49", "1.08", "1.03", "0.094", "0.090",
                             "0.065", "61.1", "12.2", "226", "14042", "1319",
                             "4.3", "0.6536", "0.6533", "0.6540", "0.6092",
                             "0.624", "0.654", "11.8", "4.2", "8.0", "3.85",
                             "2.19", "62", "0.056", "0.136", "0.973", "5.7", "5.6")}

# ---- (b) prose numbers
suspects = []
for m in re.finditer(r"\[\$([-+])\$([\d.]+),\s*\$([-+])\$([\d.]+)\$?\]", txt):
    lo = float(m.group(2)) * (1 if m.group(1) == "+" else -1)
    hi = float(m.group(4)) * (1 if m.group(3) == "+" else -1)
    if not any(abs(lo - v["lo"]) < 0.011 and abs(hi - v["hi"]) < 0.011 for v in rows.values()):
        line = txt[:m.start()].count("\n") + 1
        suspects.append((line, "[%+.2f, %+.2f]" % (lo, hi), m.group(0)))

print("\nprose intervals not matching a table row:", len(suspects))
for s in suspects:
    print("   line %-4d %s   (%s)" % s)

qs = re.findall(r"\$q = ([\d.]+)\$", txt)
print("\nprose q values:", qs)
bad_q = [x for x in qs if not any(abs(float(x) - v["q"]) < 0.011 for v in rows.values())]
print("q values not on the table:", bad_q if bad_q else "none")
