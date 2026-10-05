"""Draw the prespecified calibration and paired-effect exhibits from verified exports."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent / "followup-20261001"


def source_partition_figure(admission, verification):
    figure, axis = plt.subplots(figsize=(7.5, 7.7))
    figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    axis.set(xlim=(0, 100), ylim=(0, 100))
    axis.axis("off")

    def box(left, bottom, width, height, heading, lines, color):
        axis.add_patch(FancyBboxPatch((left, bottom), width, height,
                                     boxstyle="round,pad=0.6,rounding_size=1",
                                     facecolor=color, edgecolor="#738495", linewidth=1))
        axis.text(left + width / 2, bottom + height - 3, heading,
                  ha="center", va="top", weight="bold", fontsize=11)
        axis.text(left + width / 2, bottom + height - 8, lines,
                  ha="center", va="top", fontsize=10.5, linespacing=1.45)

    def arrow(start, finish):
        axis.add_patch(FancyArrowPatch(start, finish, arrowstyle="-|>",
                                       mutation_scale=13, color="#43596E", linewidth=1.3))

    axis.text(25, 98, "DEVELOPMENT AND DESIGN", ha="center", fontsize=11, weight="bold")
    axis.text(75, 98, "ADDITIONAL EVALUATION", ha="center", fontsize=11, weight="bold")
    box(2, 76, 46, 18, "Exposed initial evaluations",
        "Internal: 34,593 records\nPhishVN: 8,701 records\nDiagnosis → candidate design only", "#EDF1F5")
    box(52, 76, 46, 18, "Publisher 2020 benchmark",
        f"{admission['input_rows']:,} labeled source rows\nRetrospective URL-only comparison\nNo live destinations fetched", "#EDF1F5")
    arrow((25, 75), (25, 71))
    arrow((75, 75), (75, 71))
    box(2, 50, 46, 20, "Unchanged PhiUSIIL partitions",
        "Training: 166,248 records\nScaler and candidate coefficients only\nValidation: 32,695 records\nThreshold selection only", "#E5F0EF")
    box(52, 50, 46, 20, "Eligibility before predictions",
        "Valid URLs and duplicate rules\nExclude domains in retained full\nPhiUSIIL / PhishVN manifests\n"
        f"{admission['quarantined_rows']:,} rows quarantined", "#EDF1F5")
    arrow((25, 49), (25, 44))
    arrow((75, 49), (75, 44))
    box(2, 27, 46, 16, "Before external scoring",
        "Freeze candidate and threshold\nRetain unchanged 25-feature comparator\nNo external threshold selection", "#E5F0EF")
    box(52, 27, 46, 16, "Admitted benchmark",
        f"{admission['retained_rows']:,} rows; {verification['paired_uncertainty']['domain_count']:,} domains\n"
        f"Phishing: {admission['retained_class_counts']['1']:,}; legitimate: {admission['retained_class_counts']['0']:,}\n"
        "Evaluation only; not development data", "#E5F0EF")
    arrow((25, 26), (35, 21))
    arrow((75, 26), (65, 21))
    box(12, 4, 76, 16, "One paired comparison at frozen operating points",
        "All admitted rows + prescribed HTTP(S) companions\nSaved predictions → metrics and domain-cluster intervals\nSeparate arithmetic verification; no refitting or new predictions", "#DAE7EE")
    return figure


def main():
    evidence = ROOT / "verified-detection-v1"
    report = json.loads((evidence / "verification.json").read_bytes())
    if report["status"] != "verified":
        raise ValueError("Detection evidence is not verified")
    for name, expected in report["export_hashes"].items():
        if hashlib.sha256((evidence / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed verified export: {name}")
    admission_name = "population-admission-v1-attempt-2/summary.json"
    admission_content = (ROOT / admission_name).read_bytes()
    if hashlib.sha256(admission_content).hexdigest() != report["source_hashes"][admission_name]:
        raise ValueError("Changed verified population admission")
    admission = json.loads(admission_content)
    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#182D42", "text.color": "#182D42"})
    colors = {"baseline": "#68798A", "candidate": "#006D77"}
    names = {"baseline": "Unchanged 25 features", "candidate": "Transport-neutral 24 features"}
    with (evidence / "calibration-bins.csv").open() as source:
        rows = list(csv.DictReader(source))
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), gridspec_kw={"height_ratios": [1.6, 1]}, layout="constrained")
    axes[0].plot([0, 1], [0, 1], "--", color="#BABABA", linewidth=1, label="Perfect calibration")
    for position, model in enumerate(("baseline", "candidate")):
        selected = [row for row in rows if row["model"] == model]
        populated = [row for row in selected if int(row["count"]) > 0]
        axes[0].plot([float(row["mean_probability"]) for row in populated],
                     [float(row["positive_fraction"]) for row in populated],
                     "o-", color=colors[model], label=names[model], linewidth=1.5, markersize=5)
        axes[1].bar([int(row["bin"]) + (position - 0.5) * 0.37 for row in selected],
                    [int(row["count"]) for row in selected], width=0.36, color=colors[model])
    axes[0].set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025), xlabel="Mean predicted phishing probability",
                ylabel="Observed publisher-phishing fraction", title="Calibration on the admitted 2020 URL benchmark")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].set(xlabel="Fixed probability bin (width 0.1)", ylabel="Records", xticks=range(10),
                xticklabels=[f"{index / 10:.1f}" for index in range(10)])
    axes[1].text(0, -0.44, "8,622 records; observed class mixture, not deployment prevalence.\nEmpty bins have no reliability point; probability 1 belongs to the final bin.",
                 transform=axes[1].transAxes, fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(destination / f"followup-calibration.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)
    paired = report["paired_uncertainty"]
    point = paired["recall_difference"]["point"] * 100
    lower, upper = [value * 100 for value in paired["recall_difference"]["interval_97_5"]]
    figure, axis = plt.subplots(figsize=(7.5, 2.7), layout="constrained")
    axis.errorbar([point], [0], xerr=[[point - lower], [upper - point]], fmt="o", color="#006D77", capsize=6)
    axis.axvline(0, color="#68798A", linestyle="--", linewidth=1)
    axis.axvline(5, color="#B06B25", linestyle=":", linewidth=1)
    axis.set(xlim=(-7, 7), ylim=(-0.6, 0.8), yticks=[], xlabel="Recall difference, candidate minus baseline (percentage points)",
             title="Paired domain-cluster estimate at the frozen thresholds")
    axis.text(point, 0.3, f"{point:.2f} pp  [{lower:.2f}, {upper:.2f}]", ha="center", fontsize=11)
    axis.text(5, 0.57, "D target: ≥5 pp", ha="center", fontsize=9, color="#8B551F")
    axis.text(0.02, -0.4, "97.5% interval; 10,000 paired domain resamples; 6,273 domains.\nThe separate candidate FPR and scheme-invariance conditions also apply.",
              transform=axis.transAxes, fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(destination / f"followup-paired-recall.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)
    figure = source_partition_figure(admission, report)
    for extension in ("png", "pdf"):
        figure.savefig(destination / f"followup-source-partitions.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)
    manifest = {"evidence_sha256": hashlib.sha256((evidence / "verification.json").read_bytes()).hexdigest(),
                "population_admission_sha256": hashlib.sha256(admission_content).hexdigest(),
                "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.iterdir() if path.suffix in (".png", ".pdf")}}
    (destination / "detection-figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
