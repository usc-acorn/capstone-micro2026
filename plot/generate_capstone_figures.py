#!/usr/bin/env python3
"""
Generate Capstone Figures 11, 12, 13, 14, 15, and Tables 7 and 8.

Run from any directory:

    python3 plot/generate_capstone_figures.py \
        --data-dir data \
        --output-dir generated_figures

Before plotting Figure 12, create its derived sweep CSV:

    python3 plot/generate_figure12_sweep_data.py \
        --trace data/capstone_tensor3_ttv_high_cap/capstone_all_modes_trace.csv \
        --output data/figure12_sweep.csv

Dependencies:

    python3 -m pip install numpy matplotlib

Metric definitions:

* Figure 12 reads data/figure12_sweep.csv produced by
  generate_figure12_sweep_data.py.
* Figures 13 and 15 and Tables 7 and 8 use the selected model mean P_mean_mW 
(power model estimate without the upper bound applied).
* Slack is 100 * (cap - P_mean_mW) / cap.
* Figure 14 normalizes every selected frequency to that kernel's baseline.
* Table 7 normalizes every selected frequency to full bounds with K=90.
* Table 8 computes Cascade and Capstone rows from tensor3_innerprod and
  mat_sddmm. Prior-work rows are fixed values transcribed from the paper.
* Table 8's optimistic 2x and 4x throttling columns divide both frequency and
  power by two and four, respectively.
* Capstone III refers to the full-bounds controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


KERNEL_ORDER = [
    "vec_elemadd",
    "mat_elemmul",
    "tensor3_ttv",
    "tensor3_mttkrp",
    "tensor3_innerprod",
    "mat_sddmm",
    "mat_mask_tri",
    "mat_mattransmul",
]

FIGURE11_KERNEL = "tensor3_ttv"
FIGURE12_SWEEP_CSV = "figure12_sweep.csv"

FILES = {
    "bitstreams": "capstone_all_modes_bitstreams.csv",
    "selection": "capstone_all_modes_selection.json",
    "summary": "capstone_all_modes_summary.csv",
    "trace": "capstone_all_modes_trace.csv",
    "timing": "capstone_figure11_timing.csv",
}

REFERENCE_SELECTIONS = {
    "vec_elemadd": {
        "cap_mW": 350.0,
        "baseline": (439.0, 371.9284669478902),
        "capstone_i": (263.0, 210.45581162312143),
        "capstone_ii": (338.0, 271.9283524651567),
        "capstone_iii_full": (370.0, 301.65076995214366),
    },
    "mat_elemmul": {
        "cap_mW": 650.0,
        "baseline": (479.0, 770.1248008421064),
        "capstone_i": (303.0, 448.0489121019552),
        "capstone_ii": (337.0, 504.0269239503728),
        "capstone_iii_full": (374.0, 562.9699041153505),
    },
    "tensor3_ttv": {
        "cap_mW": 700.0,
        "baseline": (431.0, 702.3634282063533),
        "capstone_i": (313.0, 479.22403074436903),
        "capstone_ii": (346.0, 535.6162921346598),
        "capstone_iii_full": (384.0, 602.7803565536133),
    },
    "tensor3_mttkrp": {
        "cap_mW": 1300.0,
        "baseline": (431.0, 1384.535239540606),
        "capstone_i": (295.0, 894.308558501759),
        "capstone_ii": (333.0, 1022.7707046531465),
        "capstone_iii_full": (360.0, 1116.9229886261344),
    },
    "tensor3_innerprod": {
        "cap_mW": 750.0,
        "baseline": (481.0, 909.4408560842875),
        "capstone_i": (294.0, 510.2386353787169),
        "capstone_ii": (335.0, 586.580533709162),
        "capstone_iii_full": (366.0, 645.568474880041),
    },
    "mat_sddmm": {
        "cap_mW": 1300.0,
        "baseline": (479.0, 1488.5567792860204),
        "capstone_i": (310.0, 893.9846064196975),
        "capstone_ii": (346.0, 1011.3045395814542),
        "capstone_iii_full": (379.0, 1121.4395430051131),
    },
    "mat_mask_tri": {
        "cap_mW": 1100.0,
        "baseline": (483.0, 1262.1339001459048),
        "capstone_i": (312.0, 756.4136822472539),
        "capstone_ii": (346.0, 849.2074155721372),
        "capstone_iii_full": (385.0, 956.4124630378409),
    },
    "mat_mattransmul": {
        "cap_mW": 1000.0,
        "baseline": (483.0, 1187.1197676941672),
        "capstone_i": (308.0, 689.0239101790621),
        "capstone_ii": (342.0, 780.7355188280025),
        "capstone_iii_full": (371.0, 854.8860733490842),
    },
}

REFERENCE_FIGURE11_NORMALIZED = {
    "baseline_normalized_compile_time": 1.0,
    "capstone_i_normalized_compile_time": 0.6768948325433524,
    "capstone_ii_normalized_compile_time": 0.7514743527836063,
    "capstone_iii_normalized_compile_time": 0.760984681753263,
}

REFERENCE_FIGURE12_TRADEOFF = {
    "capstone_i": (700.0, 313.0, 479.22403074436903),
    "capstone_ii": (700.0, 346.0, 535.6162921346598),
    "capstone_iii_full": (700.0, 384.0, 602.7803565536133),
}

REFERENCE_TABLE7 = {
    ("fit_1x", 90): (1.0771502520541958, 5.455220866220657, 53),
    ("fit_2x", 90): (1.0416021191507048, 9.491983160656078, 51),
    ("fit_activity", 90): (1.0488949503198732, 8.59989584959531, 51),
    ("fit_activity_pvt", 90): (
        1.0208237853334807,
        11.82091612153797,
        50,
    ),
    ("full", 90): (1.0, 13.851293110149957, 49),
    ("full", 8): (1.0, 13.851293110149957, 8),
    ("full", 4): (1.0, 13.851293110149957, 4),
}

MAIN_MODES = [
    "baseline",
    "capstone_i",
    "capstone_ii",
    "capstone_iii_full",
]

MODE_LABELS = {
    "baseline": "Baseline",
    "capstone_i": "Cap I",
    "capstone_ii": "Cap II",
    "capstone_iii_full": "Cap III",
}

COLORS = {
    "baseline": "#fceed9",
    "capstone_i": "#f8dada",
    "capstone_ii": "#f0fedb",
    "capstone_iii_full": "#ebf6fe",
}

FIGURE12_LABELS = {
    "baseline": "Baseline (no capping)",
    "capstone_i": "Capstone I",
    "capstone_ii": "Capstone II",
    "capstone_iii_full": "Capstone III",
}

FIGURE12_MARKERS = {
    "baseline": "o",
    "capstone_i": "s",
    "capstone_ii": "^",
    "capstone_iii_full": "D",
}

TABLE7_SETTINGS = [
    ("fit_1x", 90),
    ("fit_2x", 90),
    ("fit_activity", 90),
    ("fit_activity_pvt", 90),
    ("full", 90),
    ("full", 8),
    ("full", 4),
]

TABLE7_LABELS = {
    ("fit_1x", 90): r"$1\times$ fit",
    ("fit_2x", 90): r"$2\times$ fit",
    ("fit_activity", 90): "Fit + activity",
    ("fit_activity_pvt", 90): "Fit + activity + PVT",
    ("full", 90): "Full bounds",
    ("full", 8): r"Pruned to $K\leq 8$",
    ("full", 4): r"Pruned to $K\leq 4$",
}

TABLE7_BOUNDS = {
    ("fit_1x", 90): r"$\epsilon_{e,\mathrm{fit}}$",
    ("fit_2x", 90): r"$2\epsilon_{e,\mathrm{fit}}$",
    ("fit_activity", 90): (
        r"$\epsilon_{e,\mathrm{fit}}+\epsilon_{e,\mathrm{act}}$"
    ),
    ("fit_activity_pvt", 90): (
        r"$\epsilon_{e,\mathrm{fit}}+\epsilon_{e,\mathrm{act}}"
        r"+\epsilon_{e,\mathrm{PVT}}$"
    ),
    ("full", 90): (
        r"$\epsilon_{e,\mathrm{fit}}+\epsilon_{e,\mathrm{act}}"
        r"+\epsilon_{e,\mathrm{PVT}}+\epsilon_{e,\mathrm{OOD}}$"
    ),
    ("full", 8): "Full bounds, top 8 retained",
    ("full", 4): "Full bounds, top 4 retained",
}

TABLE7_LABELS_PLAIN = {
    ("fit_1x", 90): "1× fit",
    ("fit_2x", 90): "2× fit",
    ("fit_activity", 90): "Fit + activity",
    ("fit_activity_pvt", 90): "Fit + activity + PVT",
    ("full", 90): "Full bounds",
    ("full", 8): "Pruned to K ≤ 8",
    ("full", 4): "Pruned to K ≤ 4",
}

TABLE7_BOUNDS_PLAIN = {
    ("fit_1x", 90): "ε_fit",
    ("fit_2x", 90): "2ε_fit",
    ("fit_activity", 90): "ε_fit + ε_act",
    ("fit_activity_pvt", 90): "ε_fit + ε_act + ε_PVT",
    ("full", 90): "ε_fit + ε_act + ε_PVT + ε_OOD",
    ("full", 8): "Full bounds, top 8 retained",
    ("full", 4): "Full bounds, top 4 retained",
}

TABLE8_PRIOR_ROWS: list[dict[str, Any]] = [
    {
        "compiler": "RipTide",
        "tech": "22FFL",
        "fabric": "6×6",
        "workload": "FFT",
        "cap_mW": 0.30,
        "cap_display": "0.30",
        "freq_MHz": [50.0, 25.0, 12.5],
        "power_mW": [0.24, 0.12, 0.06],
        "delta_cap_pct": [20.0, 60.0, 80.0],
        "success": ["Y", "Y", "Y"],
        "source": "paper_hardcoded",
    },
    {
        "compiler": "Snafu",
        "tech": "22FFL",
        "fabric": "6×6",
        "workload": "FFT",
        "cap_mW": 0.40,
        "cap_display": "0.40",
        "freq_MHz": [50.0, 25.0, 12.5],
        "power_mW": [0.54, 0.27, 0.135],
        "delta_cap_pct": [-35.0, 32.5, 66.25],
        "success": ["N", "Y", "Y"],
        "source": "paper_hardcoded",
    },
    {
        "compiler": "UE-CGRA",
        "tech": "28nm",
        "fabric": "8×8",
        "workload": "FFT",
        "cap_mW": 5.0,
        "cap_display": "5",
        "freq_MHz": [750.0, 325.0, 162.5],
        "power_mW": [14.0, 7.0, 3.5],
        "delta_cap_pct": [-180.0, -40.0, 30.0],
        "success": ["N", "N", "Y"],
        "source": "paper_hardcoded",
    },
    {
        "compiler": "Plasticine",
        "tech": "28nm",
        "fabric": "–",
        "workload": "Inner Prod.",
        "cap_mW": 3000.0,
        "cap_display": "3000",
        "freq_MHz": [280.0, 140.0, 70.0],
        "power_mW": [18900.0, 9450.0, 4725.0],
        "delta_cap_pct": [-530.0, -215.0, -57.5],
        "success": ["N", "N", "N"],
        "source": "paper_hardcoded",
    },
]

TABLE8_COMPUTED_MODES = [
    ("Cascade", "baseline", "12nm"),
    ("Capstone I", "capstone_i", "16nm"),
    ("Capstone II", "capstone_ii", "16nm"),
    ("Capstone III", "capstone_iii_full", "16nm"),
]

TABLE8_WORKLOADS = [
    ("tensor3_innerprod", "Inner Prod."),
    ("mat_sddmm", "SDDMM"),
]

# Signoff oracle data is not present in the all-modes dumps because
# it requires per-candidate signoff power.
FIGURE13_REFERENCE_ROWS: list[dict[str, Any]] = [
    {
        "controller": "Signoff oracle",
        "success_pct": 100.0,
        "median_delta_cap_pct": 3.13,
        "avg_norm_freq": 1.0,
        "K": "1",
        "source": "transcribed_from_paper",
    },
]


@dataclass
class RunData:
    kernel: str
    directory: Path
    run_id: str
    cap_mw: float
    selection: dict[str, Any]
    timing: dict[str, str]

    def selected(self, mode: str) -> dict[str, Any]:
        candidate = self.selection.get("selected_modes", {}).get(mode)
        if candidate is None:
            raise ValueError(f"{self.kernel}: no selected candidate for {mode}.")
        return candidate

    def table7_candidate(self, bound_mode: str, k_value: int) -> dict[str, Any] | None:
        for row in self.selection.get("table7", []):
            if str(row.get("bound_mode")) == bound_mode and to_int(row.get("K")) == k_value:
                return row.get("selected")
        return None

    def table7_retained(self, bound_mode: str, k_value: int) -> int:
        for row in self.selection.get("table7", []):
            if str(row.get("bound_mode")) == bound_mode and to_int(row.get("K")) == k_value:
                return int(row.get("retained_count", 0))
        return 0


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected a number, got {value!r}.") from error


def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected an integer, got {value!r}.") from error


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.unicode_minus"] = False


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def discover_runs(data_dir: Path) -> list[RunData]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    discovered: dict[str, RunData] = {}
    for directory in sorted(data_dir.glob("capstone_*")):
        if not directory.is_dir():
            continue
        kernel = directory.name.removeprefix("capstone_")
        # Figure 12 is created later in this script
        if kernel.endswith("_high_cap"):
            continue
        missing = [
            filename
            for filename in FILES.values()
            if not (directory / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{directory} is missing: {', '.join(missing)}"
            )

        selection_path = directory / FILES["selection"]
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        run_id = str(selection.get("run_id", "")).strip()
        if not run_id:
            raise ValueError(f"{selection_path} does not contain run_id.")

        cap_mw = to_float(selection.get("power_cap_mW"))
        if cap_mw is None or cap_mw <= 0.0:
            raise ValueError(f"{kernel}: power_cap_mW must be positive.")

        timing_rows = [
            row
            for row in read_csv_rows(directory / FILES["timing"])
            if row.get("run_id") == run_id
        ]
        if not timing_rows:
            raise ValueError(
                f"{kernel}: timing CSV has no row for run_id={run_id!r}."
            )

        # Cross-check the manifest against the rank 0 bitstream and summary rows.
        summary_rows = [
            row
            for row in read_csv_rows(directory / FILES["summary"])
            if row.get("run_id") == run_id
        ]
        bitstream_rows = [
            row
            for row in read_csv_rows(directory / FILES["bitstreams"])
            if row.get("run_id") == run_id
        ]
        for mode in MAIN_MODES:
            candidate = selection.get("selected_modes", {}).get(mode)
            if candidate is None:
                raise ValueError(f"{kernel}: manifest has no {mode} selection.")
            iteration = int(candidate["iteration"])
            if not any(
                row.get("mode") == mode
                and to_int(row.get("selected_iteration")) == iteration
                for row in summary_rows
            ):
                raise ValueError(
                    f"{kernel}: summary CSV disagrees with the {mode} "
                    f"manifest selection at iteration {iteration}."
                )
            if not any(
                row.get("mode") == mode
                and to_int(row.get("rank")) == 0
                and to_int(row.get("iteration")) == iteration
                for row in bitstream_rows
            ):
                raise ValueError(
                    f"{kernel}: bitstream CSV has no rank-0 {mode} candidate "
                    f"at iteration {iteration}."
                )

        if kernel in discovered:
            raise ValueError(f"Duplicate data directory for kernel {kernel}.")
        discovered[kernel] = RunData(
            kernel=kernel,
            directory=directory,
            run_id=run_id,
            cap_mw=float(cap_mw),
            selection=selection,
            timing=timing_rows[-1],
        )

    missing_kernels = [kernel for kernel in KERNEL_ORDER if kernel not in discovered]
    extra_kernels = sorted(set(discovered) - set(KERNEL_ORDER))
    if missing_kernels:
        raise ValueError(
            "Missing expected kernel directories: " + ", ".join(missing_kernels)
        )
    if extra_kernels:
        print(
            "warning: ignoring unrecognized kernel directories: "
            + ", ".join(extra_kernels),
            file=sys.stderr,
        )
    return [discovered[kernel] for kernel in KERNEL_ORDER]


def slack_pct(cap_mw: float, power_mw: float) -> float:
    return 100.0 * (cap_mw - power_mw) / cap_mw


def selected_metrics(runs: list[RunData]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        baseline_f = float(run.selected("baseline")["f_mhz"])
        for mode in MAIN_MODES:
            candidate = run.selected(mode)
            mean_power = float(candidate["P_mean_mW"])
            rows.append(
                {
                    "run_id": run.run_id,
                    "kernel": run.kernel,
                    "mode": mode,
                    "cap_mW": run.cap_mw,
                    "iteration": int(candidate["iteration"]),
                    "breaks": int(candidate["breaks"]),
                    "f_mhz": float(candidate["f_mhz"]),
                    "baseline_f_mhz": baseline_f,
                    "norm_freq_vs_baseline": float(candidate["f_mhz"])
                    / baseline_f,
                    "P_mean_mW": mean_power,
                    "P_upper_mW": float(candidate["P_upper_mW"]),
                    "delta_cap_pct": (
                        ""
                        if mode == "baseline"
                        else slack_pct(run.cap_mw, mean_power)
                    ),
                    "success": mean_power <= run.cap_mw + 1e-9,
                }
            )
    return rows


def plot_figure12(sweep_csv: Path, output_dir: Path) -> None:
    if not sweep_csv.is_file():
        raise FileNotFoundError(
            f"Figure 12 sweep CSV does not exist: {sweep_csv}. Generate it "
            "first with: python3 plot/generate_figure12_sweep_data.py "
            "--trace data/capstone_tensor3_ttv_high_cap/"
            "capstone_all_modes_trace.csv --output data/figure12_sweep.csv"
        )
    rows = read_csv_rows(sweep_csv)
    if not rows:
        raise ValueError(f"Figure 12 sweep CSV is empty: {sweep_csv}")

    sweep_rows = [row for row in rows if row.get("panel") == "a_cap_sweep"]
    tradeoff_rows = [row for row in rows if row.get("panel") == "b_tradeoff"]
    if not sweep_rows or not tradeoff_rows:
        raise ValueError(
            f"{sweep_csv} must contain a_cap_sweep and b_tradeoff rows."
        )

    required_modes = set(MAIN_MODES)
    sweep_modes = {row.get("mode") for row in sweep_rows}
    tradeoff_modes = {row.get("mode") for row in tradeoff_rows}
    if not required_modes.issubset(sweep_modes):
        missing = sorted(required_modes - sweep_modes)
        raise ValueError(
            "Figure 12(a) is missing modes: " + ", ".join(missing)
        )
    required_tradeoff = required_modes - {"baseline"}
    if not required_tradeoff.issubset(tradeoff_modes):
        missing = sorted(required_tradeoff - tradeoff_modes)
        raise ValueError(
            "Figure 12(b) is missing modes: " + ", ".join(missing)
        )

    fig, (ax_sweep, ax_tradeoff) = plt.subplots(
        1,
        2,
        figsize=(7.6, 2.8),
        dpi=200,
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )

    for mode in MAIN_MODES:
        mode_rows = sorted(
            [row for row in sweep_rows if row.get("mode") == mode],
            key=lambda row: float(to_float(row.get("cap_mW"))),
        )
        x_values = np.array(
            [float(to_float(row["cap_mW"])) for row in mode_rows],
            dtype=float,
        )
        y_values = np.array(
            [
                float(to_float(row["f_mhz"]))
                if parse_bool(row.get("safe_candidate_found"))
                and to_float(row.get("f_mhz")) is not None
                else np.nan
                for row in mode_rows
            ],
            dtype=float,
        )
        mask = ~np.isnan(y_values)
        ax_sweep.plot(
            x_values[mask],
            y_values[mask],
            label=FIGURE12_LABELS[mode],
            linewidth=1.35,
            marker=FIGURE12_MARKERS[mode],
            markersize=6.8,
            color="black",
            markerfacecolor=COLORS[mode],
            markeredgecolor="black",
            markeredgewidth=0.7,
        )

    ax_sweep.set_xlabel("Cap (mW)", fontsize=12)
    ax_sweep.set_ylabel("Frequency (MHz)", fontsize=12)
    ax_sweep.tick_params(axis="both", labelsize=9)
    ax_sweep.set_axisbelow(True)
    ax_sweep.yaxis.grid(True, linewidth=0.6, alpha=0.35)
    ax_sweep.spines["top"].set_visible(False)
    ax_sweep.spines["right"].set_visible(False)
    ax_sweep.legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        fontsize=8.5,
        columnspacing=0.9,
        handlelength=1.8,
    )

    for mode in ["capstone_i", "capstone_ii", "capstone_iii_full"]:
        matching = [row for row in tradeoff_rows if row.get("mode") == mode]
        if len(matching) != 1:
            raise ValueError(
                f"Figure 12(b) needs exactly one {mode} row, found "
                f"{len(matching)}."
            )
        row = matching[0]
        ax_tradeoff.scatter(
            float(to_float(row["delta_cap_pct"])),
            float(to_float(row["f_mhz"])),
            s=90,
            marker=FIGURE12_MARKERS[mode],
            facecolor=COLORS[mode],
            edgecolor="black",
            linewidth=0.9,
            label=FIGURE12_LABELS[mode],
        )

    ax_tradeoff.set_xlabel(r"Slack to cap, $\Delta$Cap (%)", fontsize=12)
    ax_tradeoff.set_ylabel("Frequency (MHz)", fontsize=12)
    ax_tradeoff.tick_params(axis="both", labelsize=9)
    ax_tradeoff.set_axisbelow(True)
    ax_tradeoff.grid(True, linewidth=0.6, alpha=0.35)
    ax_tradeoff.spines["top"].set_visible(False)
    ax_tradeoff.spines["right"].set_visible(False)
    ax_tradeoff.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        fontsize=7.0,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.6,
    )
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.20, top=0.77, wspace=0.38
    )
    save_figure(fig, output_dir / "figure12_controller_evaluation")


def plot_figure14(
    metric_rows: list[dict[str, Any]], runs: list[RunData], output_dir: Path
) -> None:
    lookup = {(row["kernel"], row["mode"]): row for row in metric_rows}
    x = np.arange(len(runs))
    width = 0.18

    fig, ax = plt.subplots(figsize=(6.2, 2.7), dpi=200)
    for index, mode in enumerate(MAIN_MODES):
        values = [
            lookup[(run.kernel, mode)]["norm_freq_vs_baseline"] for run in runs
        ]
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            label=MODE_LABELS[mode],
            color=COLORS[mode],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_ylabel("Norm. Frequency", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [run.kernel for run in runs], rotation=45, ha="right", fontsize=10
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(
        1.10,
        max(float(row["norm_freq_vs_baseline"]) for row in metric_rows) + 0.08,
    )
    ax.set_ylim(0.0, ymax)
    ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=9,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.6,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "figure14_normalized_frequency")


def plot_figure15(
    metric_rows: list[dict[str, Any]], runs: list[RunData], output_dir: Path
) -> None:
    modes = ["capstone_i", "capstone_ii", "capstone_iii_full"]
    lookup = {(row["kernel"], row["mode"]): row for row in metric_rows}
    x = np.arange(len(runs))
    width = 0.22
    all_values: list[float] = []

    fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=200)
    for index, mode in enumerate(modes):
        values = [
            float(lookup[(run.kernel, mode)]["delta_cap_pct"]) for run in runs
        ]
        all_values.extend(values)
        ax.bar(
            x + (index - 1.0) * width,
            values,
            width=width,
            label=MODE_LABELS[mode],
            color=COLORS[mode],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_ylabel(r"Slack $\Delta$Cap (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [run.kernel for run in runs], rotation=45, ha="right", fontsize=10
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymin = min(0.0, 5.0 * math.floor((min(all_values) - 2.0) / 5.0))
    ymax = max(10.0, 5.0 * math.ceil((max(all_values) + 2.0) / 5.0))
    ax.set_ylim(ymin, ymax)
    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.6,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "figure15_slack_to_cap")


def format_share(value: float | None) -> str:
    if value is None:
        return "–"
    if 0.0 < value < 1.0:
        return "< 1%"
    if math.isclose(value, 100.0, abs_tol=0.005):
        return "100%"
    return f"{value:.2f}%"


def plot_figure11(run: RunData, output_dir: Path) -> None:
    row = run.timing
    components = [
        (
            "STA (timing)",
            to_float(row.get("sta_per_iteration_s"), 0.0),
            to_float(row.get("sta_share_pct")),
        ),
        (
            "Pipelining",
            to_float(row.get("pipelining_per_iteration_s"), 0.0),
            to_float(row.get("pipelining_share_pct")),
        ),
        (
            "Capstone predictor",
            to_float(row.get("capstone_predictor_per_iteration_s"), 0.0),
            to_float(row.get("capstone_predictor_share_pct")),
        ),
        (
            "Post-PnR iter total",
            to_float(row.get("post_pnr_iteration_mean_s"), 0.0),
            100.0,
        ),
    ]
    pipeline_loop = float(
        to_float(row.get("pipeline_search_loop_total_s"), 0.0)
    )
    signoff_time = to_float(row.get("signoff_power_s"))
    normalized = [
        float(to_float(row.get("baseline_normalized_compile_time"), 1.0)),
        float(to_float(row.get("capstone_i_normalized_compile_time"), 0.0)),
        float(to_float(row.get("capstone_ii_normalized_compile_time"), 0.0)),
        float(to_float(row.get("capstone_iii_normalized_compile_time"), 0.0)),
    ]

    fig = plt.figure(figsize=(9.2, 3.25), dpi=200)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.30)
    ax_table = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])
    ax_table.set_axis_off()
    ax_table.set_xlim(0.0, 1.0)
    ax_table.set_ylim(0.0, 1.0)

    x_component, x_time, x_share = 0.03, 0.68, 0.96
    ax_table.text(x_component, 0.92, "Component", weight="bold", fontsize=14)
    ax_table.text(
        x_time,
        0.92,
        "Time / iter (s)",
        weight="bold",
        fontsize=14,
        ha="center",
    )
    ax_table.text(
        x_share, 0.92, "Share", weight="bold", fontsize=14, ha="right"
    )
    ax_table.hlines(
        [0.98, 0.86], 0.0, 1.0, color="black", linewidth=[1.8, 1.0]
    )

    for (label, seconds, share), y_value in zip(
        components, [0.77, 0.68, 0.59, 0.43]
    ):
        weight = "bold" if label == "Capstone predictor" else "normal"
        ax_table.text(
            x_component, y_value, label, fontsize=13, weight=weight
        )
        ax_table.text(
            x_time,
            y_value,
            f"{float(seconds):.2f}",
            fontsize=13,
            ha="center",
        )
        ax_table.text(
            x_share,
            y_value,
            format_share(share),
            fontsize=13,
            ha="right",
        )
    ax_table.hlines([0.51, 0.35], 0.0, 1.0, color="black", linewidth=1.0)

    ax_table.text(x_component, 0.25, "Pipeline loop", fontsize=13)
    ax_table.text(
        x_time, 0.25, f"{pipeline_loop:.2f}", fontsize=13, ha="center"
    )
    ax_table.text(x_share, 0.25, "–", fontsize=13, ha="right")
    ax_table.text(x_component, 0.15, "Signoff power", fontsize=13)
    ax_table.text(
        x_time,
        0.15,
        "not recorded" if signoff_time is None else f"{signoff_time:.2e}",
        fontsize=13,
        ha="center",
    )
    ax_table.text(x_share, 0.15, "–", fontsize=13, ha="right")
    ax_table.hlines(0.07, 0.0, 1.0, color="black", linewidth=1.8)

    labels = ["Baseline", "I", "II", "III"]
    x = np.arange(len(labels))
    ax_bar.bar(
        x,
        normalized,
        width=0.45,
        color=COLORS["capstone_iii_full"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax_bar.set_ylabel("Norm. Compile Time", fontsize=15)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=14)
    ax_bar.tick_params(axis="y", labelsize=12)
    ax_bar.set_axisbelow(True)
    ax_bar.yaxis.grid(True, linewidth=1.0, alpha=0.45)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_linewidth(1.5)
    ax_bar.spines["bottom"].set_linewidth(1.5)
    ax_bar.set_ylim(
        min(0.0, min(normalized) - 0.08),
        max(1.10, max(normalized) + 0.08),
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.08)
    save_figure(fig, output_dir / "figure11_runtime_impact")


def figure13_rows(
    metric_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    definitions = [
        ("Cascade", "baseline", "1"),
        ("Capstone I", "capstone_i", r"$\leq 4$"),
        ("Capstone II", "capstone_ii", r"$\leq 4$"),
        (r"Capstone III, $K\leq 4$", "capstone_iii_full", r"$\leq 4$"),
    ]
    result = []
    for controller, mode, k_label in definitions:
        rows = [row for row in metric_rows if row["mode"] == mode]
        successful = [row for row in rows if row["success"]]
        slack_values = [
            float(row["delta_cap_pct"])
            for row in successful
            if row["delta_cap_pct"] != ""
        ]
        result.append(
            {
                "controller": controller,
                "success_pct": 100.0
                * sum(bool(row["success"]) for row in rows)
                / len(rows),
                "median_delta_cap_pct": (
                    None
                    if not slack_values
                    else statistics.median(slack_values)
                ),
                "avg_norm_freq": statistics.mean(
                    float(row["norm_freq_vs_baseline"]) for row in rows
                ),
                "K": k_label,
                "source": "computed_from_all_modes_dumps",
            }
        )

    # Scalar Aggregate NNLS is an uncapped K=1 estimator baseline. Because it
    # does not change or stop the compiler search (its severe underprediction 
    # causes it to accept the maximally pipelined candidates), its selected 
    # candidates and aggregate controller metrics match the uncapped Cascade row.
    scalar_aggregate = dict(result[0])
    scalar_aggregate.update(
        {
            "controller": "Scalar Aggregate NNLS",
            "K": "1",
            "source": "derived_from_uncapped_cascade_selection",
        }
    )
    result.insert(1, scalar_aggregate)

    for offset, reference in enumerate(FIGURE13_REFERENCE_ROWS, start=2):
        result.insert(offset, dict(reference))
    return result


def render_table(
    column_labels: list[str],
    cell_rows: list[list[str]],
    column_widths: list[float],
    figsize: tuple[float, float],
    font_size: float,
    output_stem: Path,
    separator_before: int | list[int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.axis("off")
    table = ax.table(
        cellText=cell_rows,
        colLabels=column_labels,
        colWidths=column_widths,
        cellLoc="center",
        loc="center",
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.38)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.1)
        else:
            cell.visible_edges = ""
        if column_index == 0:
            cell.get_text().set_ha("left")

    last_row = len(cell_rows)
    for column_index in range(len(column_labels)):
        table[(last_row, column_index)].visible_edges = "B"
        table[(last_row, column_index)].set_linewidth(1.1)
    separators = (
        []
        if separator_before is None
        else [separator_before]
        if isinstance(separator_before, int)
        else separator_before
    )
    for separator_index in separators:
        table_row = separator_index + 1
        for column_index in range(len(column_labels)):
            table[(table_row, column_index)].visible_edges = "T"
            table[(table_row, column_index)].set_linewidth(0.8)

    fig.tight_layout(pad=0.2)
    save_figure(fig, output_stem)


def write_figure13(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    csv_rows = []
    visual_rows = []
    latex = [
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}lcccc@{}}",
        r"    \toprule",
        r"    \textbf{Controller} & \textbf{Success} & \textbf{Med. $\Delta$Cap (\%)} & \textbf{Avg. norm. freq.} & \textbf{$K$} \\",
        r"    \midrule",
    ]
    for row in rows:
        median = row["median_delta_cap_pct"]
        median_text = "–" if median is None else f"{median:.2f}"
        success_text = f"{row['success_pct']:.0f}%"
        controller_plain = row["controller"].replace(
            r"$K\leq 4$", "K ≤ 4"
        )
        k_plain = row["K"].replace(r"$\leq 4$", "≤ 4")
        visual_rows.append(
            [
                controller_plain,
                success_text,
                median_text,
                f"{row['avg_norm_freq']:.2f}",
                k_plain,
            ]
        )
        latex.append(
            "    "
            + f"{row['controller']} & {row['success_pct']:.0f}\\% & "
            + f"{'--' if median is None else f'{median:.2f}'} & "
            + f"{row['avg_norm_freq']:.2f} & {row['K']} \\\\"
        )
        csv_rows.append(
            {
                "controller": controller_plain,
                "success_pct": row["success_pct"],
                "median_delta_cap_pct": "" if median is None else median,
                "avg_norm_freq": row["avg_norm_freq"],
                "K": k_plain,
                "source": row["source"],
            }
        )
    latex.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \caption{Aggregate controller metrics across all $(\mathrm{kernel},\mathrm{cap})$ pairs.}",
            r"  \label{fig:aggregate-controller-metrics}",
            r"\end{figure}",
            "",
        ]
    )

    stem = output_dir / "figure13_aggregate_controller_metrics"
    render_table(
        ["Controller", "Success", "Med. ΔCap (%)", "Avg. norm. freq.", "K"],
        visual_rows,
        [0.34, 0.14, 0.22, 0.22, 0.08],
        (8.2, 2.4 + 0.25 * len(rows)),
        12,
        stem,
    )
    
    write_csv(
        stem.with_suffix(".csv"),
        csv_rows,
        [
            "controller",
            "success_pct",
            "median_delta_cap_pct",
            "avg_norm_freq",
            "K",
            "source",
        ],
    )


def table7_rows(runs: list[RunData]) -> list[dict[str, Any]]:
    result = []
    for bound_mode, k_value in TABLE7_SETTINGS:
        per_run = []
        for run in runs:
            candidate = run.table7_candidate(bound_mode, k_value)
            full_reference = run.table7_candidate("full", 90)
            if full_reference is None:
                raise ValueError(
                    f"{run.kernel}: Table 7 full-bounds K=90 row is missing."
                )
            if candidate is None:
                per_run.append(
                    {
                        "success": False,
                        "norm_freq": None,
                        "slack": None,
                        "retained": 0,
                    }
                )
                continue

            mean_power = float(candidate["P_mean_mW"])
            is_success = mean_power <= run.cap_mw + 1e-9
            per_run.append(
                {
                    "success": is_success,
                    "norm_freq": float(candidate["f_mhz"])
                    / float(full_reference["f_mhz"]),
                    "slack": (
                        slack_pct(run.cap_mw, mean_power)
                        if is_success
                        else None
                    ),
                    "retained": run.table7_retained(bound_mode, k_value),
                }
            )

        normalized = [
            row["norm_freq"]
            for row in per_run
            if row["norm_freq"] is not None
        ]
        slacks = [row["slack"] for row in per_run if row["slack"] is not None]
        result.append(
            {
                "setting": TABLE7_LABELS[(bound_mode, k_value)],
                "bound_construction": TABLE7_BOUNDS[
                    (bound_mode, k_value)
                ],
                "bound_mode": bound_mode,
                "success_pct": 100.0
                * sum(bool(row["success"]) for row in per_run)
                / len(per_run),
                "avg_norm_freq": statistics.mean(normalized),
                "median_delta_cap_pct": (
                    None if not slacks else statistics.median(slacks)
                ),
                "K": k_value,
                "min_retained_count": min(
                    int(row["retained"]) for row in per_run
                ),
            }
        )
    return result


def write_table7(rows: list[dict[str, Any]], output_dir: Path) -> None:
    visual_rows = []
    csv_rows = []
    latex = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Capstone III sensitivity to bounded-error size and candidate pruning.}",
        r"  \label{tab:capstone-iii-sensitivity}",
        r"  \small",
        r"  \begin{tabular}{@{}llcccc@{}}",
        r"    \toprule",
        r"    \textbf{Setting} & \textbf{Bound construction} & \textbf{Success} & \textbf{Avg. norm. freq.} & \textbf{Med. $\Delta$Cap (\%)} & \textbf{$K$} \\",
        r"    \midrule",
    ]
    for index, row in enumerate(rows):
        if index == 5:
            latex.append(r"    \midrule")
        median = row["median_delta_cap_pct"]
        median_text = "–" if median is None else f"{median:.2f}"
        setting_key = (row["bound_mode"], row["K"])
        setting_plain = TABLE7_LABELS_PLAIN[setting_key]
        bounds_plain = TABLE7_BOUNDS_PLAIN[setting_key]
        visual_rows.append(
            [
                setting_plain,
                bounds_plain,
                f"{row['success_pct']:.0f}%",
                f"{row['avg_norm_freq']:.2f}",
                median_text,
                str(row["K"]),
            ]
        )
        latex.append(
            "    "
            + f"{row['setting']} & {row['bound_construction']} & "
            + f"{row['success_pct']:.0f}\\% & "
            + f"{row['avg_norm_freq']:.2f} & "
            + f"{'--' if median is None else f'{median:.2f}'} & "
            + f"{row['K']} \\\\"
        )
        csv_rows.append(
            {
                "setting": setting_plain,
                "bound_construction": bounds_plain,
                "bound_mode": row["bound_mode"],
                "success_pct": row["success_pct"],
                "avg_norm_freq": row["avg_norm_freq"],
                "median_delta_cap_pct": "" if median is None else median,
                "K": row["K"],
                "min_retained_count": row["min_retained_count"],
            }
        )
    latex.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    stem = output_dir / "table7_capstone_iii_sensitivity"
    render_table(
        [
            "Setting",
            "Bound construction",
            "Success",
            "Avg. norm. freq.",
            "Med. ΔCap (%)",
            "K",
        ],
        visual_rows,
        [0.23, 0.35, 0.10, 0.14, 0.14, 0.04],
        (11.0, 3.5),
        10,
        stem,
        separator_before=5,
    )
    
    write_csv(
        stem.with_suffix(".csv"),
        csv_rows,
        [
            "setting",
            "bound_construction",
            "bound_mode",
            "success_pct",
            "avg_norm_freq",
            "median_delta_cap_pct",
            "K",
            "min_retained_count",
        ],
    )


def format_table8_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Table 8 contains a non-finite value: {value}")
    if abs(value) < 1.0 and value != 0.0:
        text = f"{value:.3f}"
    else:
        text = f"{value:.2f}"
    text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def table8_rows(runs: list[RunData]) -> list[dict[str, Any]]:
    run_by_kernel = {run.kernel: run for run in runs}
    missing = [
        kernel for kernel, _ in TABLE8_WORKLOADS if kernel not in run_by_kernel
    ]
    if missing:
        raise ValueError(
            "Table 8 is missing required kernel data: " + ", ".join(missing)
        )

    rows = [dict(row) for row in TABLE8_PRIOR_ROWS]
    for kernel, workload_label in TABLE8_WORKLOADS:
        run = run_by_kernel[kernel]
        for compiler, mode, technology in TABLE8_COMPUTED_MODES:
            candidate = run.selected(mode)
            frequency = float(candidate["f_mhz"])
            power = float(candidate["P_mean_mW"])
            if frequency <= 0.0 or power < 0.0:
                raise ValueError(
                    f"{kernel}/{mode}: Table 8 frequency must be positive and "
                    "mean power must be nonnegative."
                )
            frequencies = [
                frequency,
                frequency / 2.0,
                frequency / 4.0,
            ]
            powers = [
                power,
                power / 2.0,
                power / 4.0,
            ]
            deltas = [
                slack_pct(run.cap_mw, variant_power)
                for variant_power in powers
            ]
            successes = [
                "Y" if variant_power <= run.cap_mw + 1e-9 else "N"
                for variant_power in powers
            ]
            rows.append(
                {
                    "compiler": compiler,
                    "tech": technology,
                    "fabric": "32×16",
                    "workload": workload_label,
                    "cap_mW": run.cap_mw,
                    "cap_display": format_table8_number(run.cap_mw),
                    "freq_MHz": frequencies,
                    "power_mW": powers,
                    "delta_cap_pct": deltas,
                    "success": successes,
                    "source": run.directory.name,
                    "kernel": kernel,
                    "mode": mode,
                }
            )
    return rows


def validate_reference_dataset(
    runs: list[RunData],
    metric_rows: list[dict[str, Any]],
    figure12_csv: Path,
    output_dir: Path,
) -> None:
    """Validate the bundled public data by its numerical reference signature."""

    failures: list[str] = []

    def check_close(
        label: str,
        actual: Any,
        expected: float,
        *,
        tolerance: float = 1e-6,
    ) -> None:
        value = to_float(actual)
        if value is None or not math.isclose(
            value, expected, rel_tol=1e-9, abs_tol=tolerance
        ):
            failures.append(
                f"{label}: expected {expected:.12g}, found {actual!r}"
            )

    run_by_kernel = {run.kernel: run for run in runs}
    metric_by_key = {
        (str(row["kernel"]), str(row["mode"])): row for row in metric_rows
    }
    for kernel, expected in REFERENCE_SELECTIONS.items():
        run = run_by_kernel.get(kernel)
        if run is None:
            failures.append(f"missing reference kernel {kernel}")
            continue
        check_close(f"{kernel} cap", run.cap_mw, expected["cap_mW"])
        for mode in MAIN_MODES:
            row = metric_by_key.get((kernel, mode))
            if row is None:
                failures.append(f"{kernel}/{mode}: missing selected metric")
                continue
            expected_frequency, expected_power = expected[mode]
            check_close(
                f"{kernel}/{mode} frequency",
                row["f_mhz"],
                expected_frequency,
            )
            check_close(
                f"{kernel}/{mode} mean power",
                row["P_mean_mW"],
                expected_power,
            )

    figure11_run = run_by_kernel.get(FIGURE11_KERNEL)
    if figure11_run is None:
        failures.append(f"missing Figure 11 kernel {FIGURE11_KERNEL}")
    else:
        for field, expected in REFERENCE_FIGURE11_NORMALIZED.items():
            check_close(
                f"Figure 11 {field}",
                figure11_run.timing.get(field),
                expected,
            )

    sweep_rows = read_csv_rows(figure12_csv)
    if len(sweep_rows) != 39:
        failures.append(
            f"Figure 12 sweep: expected 39 rows, found {len(sweep_rows)}"
        )
    tradeoff_rows = {
        str(row.get("mode")): row
        for row in sweep_rows
        if row.get("panel") == "b_tradeoff"
    }
    for mode, expected in REFERENCE_FIGURE12_TRADEOFF.items():
        row = tradeoff_rows.get(mode)
        if row is None:
            failures.append(f"Figure 12 tradeoff: missing {mode}")
            continue
        expected_cap, expected_frequency, expected_power = expected
        check_close(f"Figure 12 {mode} cap", row.get("cap_mW"), expected_cap)
        check_close(
            f"Figure 12 {mode} frequency",
            row.get("f_mhz"),
            expected_frequency,
        )
        check_close(
            f"Figure 12 {mode} mean power",
            row.get("P_mean_mW"),
            expected_power,
        )

    computed_table7 = {
        (str(row["bound_mode"]), int(row["K"])): row
        for row in table7_rows(runs)
    }
    for key, expected in REFERENCE_TABLE7.items():
        row = computed_table7.get(key)
        if row is None:
            failures.append(f"Table 7: missing bound mode {key}")
            continue
        expected_norm, expected_slack, expected_retained = expected
        check_close(
            f"Table 7 {key} average normalized frequency",
            row["avg_norm_freq"],
            expected_norm,
        )
        check_close(
            f"Table 7 {key} median slack",
            row["median_delta_cap_pct"],
            expected_slack,
        )
        if int(row["min_retained_count"]) != expected_retained:
            failures.append(
                f"Table 7 {key} retained count: expected "
                f"{expected_retained}, found {row['min_retained_count']}"
            )

    expected_outputs = [
        "figure11_runtime_impact.pdf",
        "figure11_runtime_impact.png",
        "figure12_controller_evaluation.pdf",
        "figure12_controller_evaluation.png",
        "figure13_aggregate_controller_metrics.csv",
        "figure13_aggregate_controller_metrics.pdf",
        "figure13_aggregate_controller_metrics.png",
        "figure14_normalized_frequency.pdf",
        "figure14_normalized_frequency.png",
        "figure15_slack_to_cap.pdf",
        "figure15_slack_to_cap.png",
        "generation_manifest.json",
        "selected_metrics.csv",
        "table7_capstone_iii_sensitivity.csv",
        "table7_capstone_iii_sensitivity.pdf",
        "table7_capstone_iii_sensitivity.png",
        "table8_prior_cgra_capability.csv",
        "table8_prior_cgra_capability.pdf",
        "table8_prior_cgra_capability.png",
    ]
    missing_outputs = [
        filename
        for filename in expected_outputs
        if not (output_dir / filename).is_file()
    ]
    if missing_outputs:
        failures.append(
            "missing generated outputs: " + ", ".join(missing_outputs)
        )

    if failures:
        raise ValueError(
            "Reference validation failed:\n  - " + "\n  - ".join(failures)
        )

    print(
        "REFERENCE VALIDATION: PASS — bundled controller data matches "
        "the reference signature."
    )


def table8_triple(values: Sequence[float]) -> str:
    return " | ".join(format_table8_number(float(value)) for value in values)


def table8_success_triple(values: Sequence[str]) -> str:
    return " | ".join(str(value) for value in values)


def table8_latex_text(value: str) -> str:
    return (
        value.replace("×", r"$\times$")
        .replace("–", "--")
        .replace("_", r"\_")
    )


def table8_latex_triple(values: Sequence[float]) -> str:
    return (
        "$"
        + r" \mid ".join(
            format_table8_number(float(value)) for value in values
        )
        + "$"
    )


def table8_latex_success(values: Sequence[str]) -> str:
    return "$" + r" \mid ".join(str(value) for value in values) + "$"


def write_table8(rows: list[dict[str, Any]], output_dir: Path) -> None:
    visual_rows = []
    csv_rows = []
    latex = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Capability-oriented comparison with prior CGRA compilers under target power caps.}",
        r"  \label{tab:prior-cgra-capability}",
        r"  \scriptsize",
        r"  \begin{tabular}{@{}llllrllll@{}}",
        r"    \toprule",
        r"    \textbf{Compiler} & \textbf{Tech.} & \textbf{Fabric} & \textbf{Workload} & \textbf{Cap (mW)} & \textbf{Freq. (MHz)} & \textbf{Power (mW)} & \textbf{$\Delta$Cap (\%)} & \textbf{Success (orig$\mid$2$\times\mid$4$\times$)} \\",
        r"    \midrule",
    ]

    for index, row in enumerate(rows):
        if index in {4, 8}:
            latex.append(r"    \midrule")
        freq_text = table8_triple(row["freq_MHz"])
        power_text = table8_triple(row["power_mW"])
        delta_text = table8_triple(row["delta_cap_pct"])
        success_text = table8_success_triple(row["success"])
        visual_rows.append(
            [
                row["compiler"],
                row["tech"],
                row["fabric"],
                row["workload"],
                row["cap_display"],
                freq_text,
                power_text,
                delta_text,
                success_text,
            ]
        )
        latex.append(
            "    "
            + " & ".join(
                [
                    table8_latex_text(row["compiler"]),
                    table8_latex_text(row["tech"]),
                    table8_latex_text(row["fabric"]),
                    table8_latex_text(row["workload"]),
                    row["cap_display"],
                    table8_latex_triple(row["freq_MHz"]),
                    table8_latex_triple(row["power_mW"]),
                    table8_latex_triple(row["delta_cap_pct"]),
                    table8_latex_success(row["success"]),
                ]
            )
            + r" \\"
        )
        csv_rows.append(
            {
                "compiler": row["compiler"],
                "tech": row["tech"],
                "fabric": row["fabric"],
                "workload": row["workload"],
                "cap_mW": row["cap_mW"],
                "freq_MHz_orig": row["freq_MHz"][0],
                "freq_MHz_2x": row["freq_MHz"][1],
                "freq_MHz_4x": row["freq_MHz"][2],
                "power_mW_orig": row["power_mW"][0],
                "power_mW_2x": row["power_mW"][1],
                "power_mW_4x": row["power_mW"][2],
                "delta_cap_pct_orig": row["delta_cap_pct"][0],
                "delta_cap_pct_2x": row["delta_cap_pct"][1],
                "delta_cap_pct_4x": row["delta_cap_pct"][2],
                "success_orig": row["success"][0],
                "success_2x": row["success"][1],
                "success_4x": row["success"][2],
                "source": row["source"],
            }
        )

    latex.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    stem = output_dir / "table8_prior_cgra_capability"
    render_table(
        [
            "Compiler",
            "Tech.",
            "Fabric",
            "Workload",
            "Cap (mW)",
            "Freq. (MHz)",
            "Power (mW)",
            "ΔCap (%)",
            "Success (orig | 2× | 4×)",
        ],
        visual_rows,
        [0.11, 0.07, 0.07, 0.11, 0.08, 0.16, 0.19, 0.11, 0.10],
        (15.5, 5.6),
        8.0,
        stem,
        separator_before=[4, 8],
    )
    
    write_csv(
        stem.with_suffix(".csv"),
        csv_rows,
        [
            "compiler",
            "tech",
            "fabric",
            "workload",
            "cap_mW",
            "freq_MHz_orig",
            "freq_MHz_2x",
            "freq_MHz_4x",
            "power_mW_orig",
            "power_mW_2x",
            "power_mW_4x",
            "delta_cap_pct_orig",
            "delta_cap_pct_2x",
            "delta_cap_pct_4x",
            "success_orig",
            "success_2x",
            "success_4x",
            "source",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Folder containing capstone_<kernel>/ directories (default: data next to script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: generated_figures next to script).",
    )
    parser.add_argument(
        "--validate-reference",
        action="store_true",
        help=(
            "Validate the bundled pre-generated data and expected outputs "
            "against the artifact's numerical reference signature."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = (
        args.data_dir.resolve()
        if args.data_dir is not None
        else script_dir / "data"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else script_dir / "generated_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    runs = discover_runs(data_dir)
    metrics = selected_metrics(runs)
    write_csv(
        output_dir / "selected_metrics.csv",
        metrics,
        [
            "run_id",
            "kernel",
            "mode",
            "cap_mW",
            "iteration",
            "breaks",
            "f_mhz",
            "baseline_f_mhz",
            "norm_freq_vs_baseline",
            "P_mean_mW",
            "P_upper_mW",
            "delta_cap_pct",
            "success",
        ],
    )

    figure11_run = next(
        (run for run in runs if run.kernel == FIGURE11_KERNEL), None
    )
    if figure11_run is None:
        raise ValueError(
            f"Figure 11 kernel {FIGURE11_KERNEL!r} was not discovered."
        )
    plot_figure11(figure11_run, output_dir)
    figure12_csv = data_dir / FIGURE12_SWEEP_CSV
    plot_figure12(figure12_csv, output_dir)
    write_figure13(figure13_rows(metrics), output_dir)
    plot_figure14(metrics, runs, output_dir)
    plot_figure15(metrics, runs, output_dir)
    write_table7(table7_rows(runs), output_dir)
    write_table8(table8_rows(runs), output_dir)

    provenance = {
        "data_dir": data_dir.name,
        "metric_power_source": "P_mean_mW",
        "figure11_kernel": FIGURE11_KERNEL,
        "figure12_sweep_csv": figure12_csv.name,
        "figure13_reference_rows": [
            {
                "controller": "Scalar Aggregate NNLS",
                "source": "derived_from_uncapped_cascade_selection",
            },
            *[
                {
                    "controller": row["controller"],
                    "source": row["source"],
                }
                for row in FIGURE13_REFERENCE_ROWS
            ],
        ],
        "capstone_iii_main_mode": "capstone_iii_full",
        "table8": {
            "computed_kernels": [
                kernel for kernel, _ in TABLE8_WORKLOADS
            ],
            "prior_rows": [
                row["compiler"] for row in TABLE8_PRIOR_ROWS
            ],
            "throttling_divisors": [1, 2, 4],
        },
        "runs": [
            {
                "kernel": run.kernel,
                "run_id": run.run_id,
                "cap_mW": run.cap_mw,
                "directory": run.directory.name,
            }
            for run in runs
        ],
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    if args.validate_reference:
        validate_reference_dataset(
            runs,
            metrics,
            figure12_csv,
            output_dir,
        )

    print(f"Read {len(runs)} kernel result directories from {data_dir}")
    print(
        f"Generated Figures 11, 12, 13, 14, 15, and Tables 7 and 8 in "
        f"{output_dir}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
