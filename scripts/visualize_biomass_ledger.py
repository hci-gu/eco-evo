#!/usr/bin/env python3
"""Create a readable HTML report from a Mareld biomass ledger .npz file."""
from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path

import numpy as np


def _setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mareld")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _decode_labels(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _pretty_label(value):
    return str(value).replace("_", " ")


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _top_indices(values, n):
    flat = np.asarray(values).ravel()
    if flat.size == 0:
        return []
    order = np.argsort(flat)[::-1]
    return [int(i) for i in order[:n] if flat[i] > 0.0]


def _save_heatmap(plt, matrix, row_labels, col_labels, title, cbar_label, path):
    fig_w = max(7.0, 0.55 * len(col_labels) + 2.5)
    fig_h = max(4.5, 0.42 * len(row_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Prey")
    ax.set_ylabel("Predator")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([_pretty_label(x) for x in col_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([_pretty_label(x) for x in row_labels])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    if matrix.size and matrix.shape[0] * matrix.shape[1] <= 120:
        threshold = float(np.nanmax(matrix)) * 0.55 if np.nanmax(matrix) > 0 else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = float(matrix[i, j])
                if value <= 0.0:
                    continue
                color = "white" if value > threshold else "black"
                ax.text(j, i, f"{value:.2g}", ha="center", va="center",
                        color=color, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_stacked_barh(plt, matrix, row_labels, stack_labels, title, xlabel, path,
                       max_stacks=10):
    totals = matrix.sum(axis=1)
    keep_rows = [i for i, total in enumerate(totals) if total > 0.0]
    if not keep_rows:
        _save_empty_plot(plt, title, "No nonzero flows recorded.", path)
        return

    data = matrix[keep_rows]
    labels = [row_labels[i] for i in keep_rows]
    stack_totals = data.sum(axis=0)
    top = _top_indices(stack_totals, max_stacks)
    if not top:
        _save_empty_plot(plt, title, "No nonzero flows recorded.", path)
        return
    other = [i for i in range(data.shape[1]) if i not in set(top)]

    plot_data = data[:, top]
    plot_labels = [stack_labels[i] for i in top]
    if other:
        other_values = data[:, other].sum(axis=1, keepdims=True)
        if float(other_values.sum()) > 0.0:
            plot_data = np.concatenate([plot_data, other_values], axis=1)
            plot_labels.append("other")

    fig_h = max(4.0, 0.46 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), constrained_layout=True)
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    cmap = plt.get_cmap("tab20")
    for k in range(plot_data.shape[1]):
        vals = plot_data[:, k]
        ax.barh(y, vals, left=left, label=_pretty_label(plot_labels[k]),
                color=cmap(k % 20))
        left += vals
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_label(x) for x in labels])
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_lines(plt, series, labels, title, ylabel, path):
    if not series:
        _save_empty_plot(plt, title, "No nonzero flows recorded.", path)
        return
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for y, label in zip(series, labels):
        ax.plot(np.arange(len(y)), y, label=_pretty_label(label), linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Tick")
    ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_normalized_biomass(plt, biomass_totals, initial_biomass,
                             species_labels, path):
    fig, ax = plt.subplots(figsize=(11.5, 6), constrained_layout=True)
    ticks = np.arange(biomass_totals.shape[0])
    for i, species in enumerate(species_labels):
        if initial_biomass[i] <= 0.0:
            continue
        pct = biomass_totals[:, i] / initial_biomass[i] * 100.0
        ax.plot(ticks, pct, label=_pretty_label(species), linewidth=1.8)
    ax.axhline(100.0, color="#9aa6b2", linewidth=1.0, linestyle="--")
    ax.set_title("Population Biomass Over Time")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Biomass (% of start)")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_decline_detail(plt, biomass_pct, loss_by_source, source_labels,
                         species_label, path, top_sources=5):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
        constrained_layout=True,
    )
    ticks = np.arange(len(biomass_pct))
    ax_top.plot(ticks, biomass_pct, color="#1f4e79", linewidth=2.2)
    ax_top.axhline(100.0, color="#9aa6b2", linewidth=1.0, linestyle="--")
    ax_top.set_title(f"{_pretty_label(species_label)} Biomass Decline")
    ax_top.set_ylabel("Biomass (% of start)")
    ax_top.set_ylim(bottom=0.0)

    source_totals = loss_by_source.sum(axis=0)
    top = _top_indices(source_totals, top_sources)
    if top:
        data = loss_by_source[:, top].T
        labels = [_pretty_label(source_labels[i]) for i in top]
        other = [i for i in range(loss_by_source.shape[1]) if i not in set(top)]
        if other:
            other_values = loss_by_source[:, other].sum(axis=1)
            if float(other_values.sum()) > 0.0:
                data = np.vstack([data, other_values])
                labels.append("other")
        ax_bottom.stackplot(ticks, data, labels=labels, alpha=0.92)
        ax_bottom.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    else:
        ax_bottom.text(0.5, 0.5, "No logged loss causes for this species.",
                       transform=ax_bottom.transAxes, ha="center", va="center")
    ax_bottom.set_xlabel("Tick")
    ax_bottom.set_ylabel("Loss per tick (tonnes)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_dying_species_gain_detail(plt, biomass_pct, gain_by_source,
                                    source_labels, species_label, path,
                                    top_sources=5):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
        constrained_layout=True,
    )
    ticks = np.arange(len(biomass_pct))
    ax_top.plot(ticks, biomass_pct, color="#1f4e79", linewidth=2.2)
    ax_top.axhline(100.0, color="#9aa6b2", linewidth=1.0, linestyle="--")
    ax_top.set_title(f"{_pretty_label(species_label)} Biomass Gains While Declining")
    ax_top.set_ylabel("Biomass (% of start)")
    ax_top.set_ylim(bottom=0.0)

    source_totals = gain_by_source.sum(axis=0)
    top = _top_indices(source_totals, top_sources)
    if top:
        data = gain_by_source[:, top].T
        labels = [_pretty_label(source_labels[i]) for i in top]
        other = [i for i in range(gain_by_source.shape[1]) if i not in set(top)]
        if other:
            other_values = gain_by_source[:, other].sum(axis=1)
            if float(other_values.sum()) > 0.0:
                data = np.vstack([data, other_values])
                labels.append("other")
        ax_bottom.stackplot(ticks, data, labels=labels, alpha=0.92)
        ax_bottom.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    else:
        ax_bottom.text(0.5, 0.5, "No logged biomass gains for this species.",
                       transform=ax_bottom.transAxes, ha="center", va="center")
    ax_bottom.set_xlabel("Tick")
    ax_bottom.set_ylabel("Gain per tick (tonnes)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_action_detail(plt, category_pct, detailed_pct, category_labels,
                        action_labels, species_label, path, top_actions=8):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
        constrained_layout=True,
    )
    ticks = np.arange(category_pct.shape[0])
    ax_top.stackplot(
        ticks,
        category_pct.T,
        labels=[_pretty_label(x) for x in category_labels],
        alpha=0.92,
    )
    ax_top.set_title(f"{_pretty_label(species_label)} Action Categories")
    ax_top.set_ylabel("Biomass-weighted decisions (%)")
    ax_top.set_ylim(0.0, 100.0)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    totals = detailed_pct.sum(axis=0)
    top = _top_indices(totals, top_actions)
    if top:
        data = detailed_pct[:, top].T
        labels = [_pretty_label(action_labels[i]) for i in top]
        other = [i for i in range(detailed_pct.shape[1]) if i not in set(top)]
        if other:
            other_values = detailed_pct[:, other].sum(axis=1)
            if float(other_values.sum()) > 0.0:
                data = np.vstack([data, other_values])
                labels.append("other")
        ax_bottom.stackplot(ticks, data, labels=labels, alpha=0.92)
        ax_bottom.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    else:
        ax_bottom.text(0.5, 0.5, "No logged actions for this species.",
                       transform=ax_bottom.transAxes, ha="center", va="center")
    ax_bottom.set_xlabel("Tick")
    ax_bottom.set_ylabel("Detailed decisions (%)")
    ax_bottom.set_ylim(0.0, 100.0)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_empty_plot(plt, title, message, path):
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "species", "gain_tonnes", "loss_tonnes", "net_tonnes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _html_table(headers, rows, max_rows=None):
    shown = rows[:max_rows] if max_rows is not None else rows
    parts = ["<table>", "<thead><tr>"]
    for h in headers:
        parts.append(f"<th>{html.escape(h)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in shown:
        parts.append("<tr>")
        for value in row:
            parts.append(f"<td>{html.escape(str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if max_rows is not None and len(rows) > max_rows:
        parts.append(f"<p class=\"muted\">Showing {max_rows} of {len(rows)} rows.</p>")
    return "\n".join(parts)


def _fmt(value):
    value = float(value)
    if abs(value) >= 1000.0:
        return f"{value:,.1f}"
    if abs(value) >= 10.0:
        return f"{value:.2f}"
    return f"{value:.4g}"


def build_report(ledger_path, outdir, top_n=10):
    plt = _setup_matplotlib()
    outdir = _ensure_dir(Path(outdir))
    figures_dir = _ensure_dir(outdir / "figures")

    with np.load(ledger_path, allow_pickle=False) as data:
        predation = np.asarray(data["predation_tonnes"], dtype=np.float64)
        assimilation = np.asarray(data["assimilation_energy"], dtype=np.float64)
        gains = np.asarray(data["biomass_gain_by_source"], dtype=np.float64)
        losses = np.asarray(data["biomass_loss_by_sink"], dtype=np.float64)
        dm_ids = _decode_labels(data["dm_ids"])
        fg_ids = _decode_labels(data["fg_ids"])
        source_labels = _decode_labels(data["source_labels"])
        initial_biomass = (
            np.asarray(data["initial_biomass"], dtype=np.float64)
            if "initial_biomass" in data.files else np.asarray([], dtype=np.float64)
        )
        biomass_totals = (
            np.asarray(data["biomass_totals"], dtype=np.float64)
            if "biomass_totals" in data.files else np.asarray([], dtype=np.float64)
        )
        action_percentages = (
            np.asarray(data["action_percentages"], dtype=np.float64)
            if "action_percentages" in data.files else np.asarray([], dtype=np.float64)
        )
        action_category_percentages = (
            np.asarray(data["action_category_percentages"], dtype=np.float64)
            if "action_category_percentages" in data.files else np.asarray([], dtype=np.float64)
        )
        action_labels = (
            _decode_labels(data["action_labels"])
            if "action_labels" in data.files else []
        )
        action_category_labels = (
            _decode_labels(data["action_category_labels"])
            if "action_category_labels" in data.files else []
        )

    total_predation = predation.sum(axis=0) if predation.size else np.zeros((len(dm_ids), len(fg_ids)))
    total_assimilation = assimilation.sum(axis=0) if assimilation.size else np.zeros_like(total_predation)
    total_gains = gains.sum(axis=0) if gains.size else np.zeros((len(fg_ids), len(source_labels)))
    total_losses = losses.sum(axis=0) if losses.size else np.zeros((len(fg_ids), len(source_labels)))

    gain_totals = total_gains.sum(axis=1)
    loss_totals = total_losses.sum(axis=1)
    net_totals = gain_totals - loss_totals
    has_biomass_totals = (
        initial_biomass.shape == (len(fg_ids),)
        and biomass_totals.ndim == 2
        and biomass_totals.shape[1] == len(fg_ids)
        and biomass_totals.shape[0] > 0
    )
    if has_biomass_totals:
        final_biomass = biomass_totals[-1]
        end_pct = np.divide(
            final_biomass,
            initial_biomass,
            out=np.zeros_like(final_biomass, dtype=np.float64),
            where=initial_biomass > 0.0,
        ) * 100.0
        decline_pct = np.maximum(0.0, 100.0 - end_pct)
    else:
        final_biomass = np.zeros(len(fg_ids), dtype=np.float64)
        end_pct = np.zeros(len(fg_ids), dtype=np.float64)
        decline_pct = np.zeros(len(fg_ids), dtype=np.float64)
    has_action_data = (
        action_percentages.ndim == 3
        and action_percentages.shape[1] == len(dm_ids)
        and action_percentages.shape[2] == len(action_labels)
        and action_category_percentages.ndim == 3
        and action_category_percentages.shape[1] == len(dm_ids)
        and action_category_percentages.shape[2] == len(action_category_labels)
        and action_percentages.shape[0] > 0
    )

    summary_rows = []
    for rank, idx in enumerate(np.argsort(np.abs(net_totals))[::-1], start=1):
        summary_rows.append({
            "rank": rank,
            "species": fg_ids[int(idx)],
            "gain_tonnes": f"{float(gain_totals[idx]):.10g}",
            "loss_tonnes": f"{float(loss_totals[idx]):.10g}",
            "net_tonnes": f"{float(net_totals[idx]):.10g}",
        })
    _write_summary_csv(outdir / "species_flow_summary.csv", summary_rows)

    paths = {}
    if has_biomass_totals:
        paths["normalized_biomass"] = figures_dir / "normalized_biomass.png"
        _save_normalized_biomass(
            plt, biomass_totals, initial_biomass, fg_ids,
            paths["normalized_biomass"],
        )

    paths["predation_heatmap"] = figures_dir / "predation_heatmap.png"
    _save_heatmap(
        plt, total_predation, dm_ids, fg_ids,
        "Total Prey Biomass Consumed",
        "Tonnes consumed",
        paths["predation_heatmap"],
    )

    paths["assimilation_heatmap"] = figures_dir / "assimilation_heatmap.png"
    _save_heatmap(
        plt, total_assimilation, dm_ids, fg_ids,
        "Total Assimilated Energy by Diet Source",
        "Energy units",
        paths["assimilation_heatmap"],
    )

    paths["diet_bars"] = figures_dir / "diet_composition.png"
    _save_stacked_barh(
        plt, total_predation, dm_ids, fg_ids,
        "Diet Composition by Predator",
        "Tonnes consumed",
        paths["diet_bars"],
        max_stacks=top_n,
    )

    paths["loss_bars"] = figures_dir / "loss_composition.png"
    _save_stacked_barh(
        plt, total_losses, fg_ids, source_labels,
        "Biomass Losses by Cause",
        "Tonnes lost",
        paths["loss_bars"],
        max_stacks=top_n,
    )

    paths["gain_bars"] = figures_dir / "gain_composition.png"
    _save_stacked_barh(
        plt, total_gains, fg_ids, source_labels,
        "Biomass Gains by Source",
        "Tonnes gained",
        paths["gain_bars"],
        max_stacks=top_n,
    )

    flow_totals = total_predation.ravel()
    top_flows = _top_indices(flow_totals, top_n)
    flow_series = []
    flow_labels = []
    for flat_idx in top_flows:
        i, j = np.unravel_index(flat_idx, total_predation.shape)
        flow_series.append(predation[:, i, j])
        flow_labels.append(f"{dm_ids[i]} eats {fg_ids[j]}")
    paths["predation_lines"] = figures_dir / "top_predation_timeseries.png"
    _save_lines(
        plt, flow_series, flow_labels,
        f"Top {min(top_n, len(flow_series))} Predation Flows Through Time",
        "Tonnes per tick",
        paths["predation_lines"],
    )

    source_gain_totals = total_gains.ravel()
    top_gain_flows = _top_indices(source_gain_totals, top_n)
    gain_series = []
    gain_labels = []
    for flat_idx in top_gain_flows:
        i, j = np.unravel_index(flat_idx, total_gains.shape)
        gain_series.append(gains[:, i, j])
        gain_labels.append(f"{fg_ids[i]} from {source_labels[j]}")
    paths["gain_lines"] = figures_dir / "top_gain_timeseries.png"
    _save_lines(
        plt, gain_series, gain_labels,
        f"Top {min(top_n, len(gain_series))} Biomass Gain Sources Through Time",
        "Tonnes per tick",
        paths["gain_lines"],
    )

    action_detail_paths = []
    if has_action_data:
        for i, dm_id in enumerate(dm_ids):
            action_path = figures_dir / f"action_decisions_{dm_id}.png"
            _save_action_detail(
                plt,
                action_category_percentages[:, i, :],
                action_percentages[:, i, :],
                action_category_labels,
                action_labels,
                dm_id,
                action_path,
                top_actions=top_n,
            )
            mean_categories = action_category_percentages[:, i, :].mean(axis=0)
            action_detail_paths.append((dm_id, action_path, mean_categories))

    extinction_detail_paths = []
    if has_biomass_totals:
        eligible = [i for i in np.argsort(decline_pct)[::-1]
                    if initial_biomass[i] > 0.0 and decline_pct[i] > 0.0]
        for idx in eligible[:3]:
            pct_series = biomass_totals[:, idx] / initial_biomass[idx] * 100.0
            loss_detail_path = figures_dir / f"decline_detail_{fg_ids[idx]}.png"
            _save_decline_detail(
                plt,
                pct_series,
                losses[:, idx, :],
                source_labels,
                fg_ids[idx],
                loss_detail_path,
                top_sources=top_n,
            )
            gain_detail_path = figures_dir / f"decline_gain_detail_{fg_ids[idx]}.png"
            _save_dying_species_gain_detail(
                plt,
                pct_series,
                gains[:, idx, :],
                source_labels,
                fg_ids[idx],
                gain_detail_path,
                top_sources=top_n,
            )
            loss_source_totals = losses[:, idx, :].sum(axis=0)
            gain_source_totals = gains[:, idx, :].sum(axis=0)
            top_causes = [
                (_pretty_label(source_labels[src_idx]), loss_source_totals[src_idx])
                for src_idx in _top_indices(loss_source_totals, 5)
            ]
            top_gain_sources = [
                (_pretty_label(source_labels[src_idx]), gain_source_totals[src_idx])
                for src_idx in _top_indices(gain_source_totals, 5)
            ]
            extinction_detail_paths.append((
                idx,
                loss_detail_path,
                gain_detail_path,
                top_causes,
                top_gain_sources,
            ))

    species_table_rows = []
    for row in summary_rows:
        species_table_rows.append([
            row["rank"],
            _pretty_label(row["species"]),
            _fmt(row["gain_tonnes"]),
            _fmt(row["loss_tonnes"]),
            _fmt(row["net_tonnes"]),
        ])

    biomass_table_rows = []
    if has_biomass_totals:
        order = np.argsort(decline_pct)[::-1]
        for rank, idx in enumerate(order, start=1):
            biomass_table_rows.append([
                rank,
                _pretty_label(fg_ids[int(idx)]),
                _fmt(initial_biomass[idx]),
                _fmt(final_biomass[idx]),
                f"{float(end_pct[idx]):.2f}%",
                f"{float(decline_pct[idx]):.2f}%",
            ])

    top_pred_rows = []
    for rank, flat_idx in enumerate(top_flows, start=1):
        i, j = np.unravel_index(flat_idx, total_predation.shape)
        top_pred_rows.append([
            rank,
            _pretty_label(dm_ids[i]),
            _pretty_label(fg_ids[j]),
            _fmt(total_predation[i, j]),
            _fmt(total_assimilation[i, j]),
        ])

    ticks = int(predation.shape[0]) if predation.ndim else 0
    total_consumed = float(total_predation.sum())
    total_assim_energy = float(total_assimilation.sum())
    total_gain = float(total_gains.sum())
    total_loss = float(total_losses.sum())

    report_path = outdir / "report.html"
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #17202a; background: #f6f7f9; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
h1 { margin: 0 0 4px; font-size: 30px; }
h2 { margin-top: 34px; border-top: 1px solid #d7dce2; padding-top: 22px; }
.muted { color: #647080; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 22px 0; }
.card { background: white; border: 1px solid #dfe4ea; border-radius: 8px; padding: 16px; }
.card .label { color: #647080; font-size: 13px; }
.card .value { font-size: 24px; font-weight: 650; margin-top: 4px; }
.figure { background: white; border: 1px solid #dfe4ea; border-radius: 8px; padding: 14px; margin: 16px 0; }
.figure img { width: 100%; height: auto; display: block; }
table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #dfe4ea; border-radius: 8px; overflow: hidden; }
th, td { padding: 9px 11px; border-bottom: 1px solid #e8edf2; text-align: left; font-size: 14px; }
th { background: #edf1f5; font-weight: 650; }
code { background: #e9eef4; padding: 2px 5px; border-radius: 4px; }
"""
    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Mareld Biomass Ledger Report</title>",
        f"<style>{css}</style>",
        "</head><body><main>",
        "<h1>Mareld Biomass Ledger Report</h1>",
        f"<p class=\"muted\">Source: <code>{html.escape(str(ledger_path))}</code></p>",
        "<div class=\"cards\">",
        f"<div class=\"card\"><div class=\"label\">Ticks</div><div class=\"value\">{ticks}</div></div>",
        f"<div class=\"card\"><div class=\"label\">Species</div><div class=\"value\">{len(fg_ids)}</div></div>",
        f"<div class=\"card\"><div class=\"label\">Consumed Biomass</div><div class=\"value\">{_fmt(total_consumed)}</div></div>",
        f"<div class=\"card\"><div class=\"label\">Assimilated Energy</div><div class=\"value\">{_fmt(total_assim_energy)}</div></div>",
        f"<div class=\"card\"><div class=\"label\">Logged Biomass Gains</div><div class=\"value\">{_fmt(total_gain)}</div></div>",
        f"<div class=\"card\"><div class=\"label\">Logged Biomass Losses</div><div class=\"value\">{_fmt(total_loss)}</div></div>",
        "</div>",
        "<h2>Species Balance</h2>",
        _html_table(["Rank", "Species", "Gains", "Losses", "Net"], species_table_rows),
    ]
    if has_biomass_totals:
        rel = paths["normalized_biomass"].relative_to(outdir)
        html_parts.extend([
            "<h2>Population Percentages</h2>",
            "<p class=\"muted\">Each line shows total biomass as a percentage of that species' starting biomass.</p>",
            "<div class=\"figure\">",
            f"<img src=\"{html.escape(str(rel))}\" alt=\"Normalized biomass over time\">",
            "</div>",
        ])
    else:
        html_parts.extend([
            "<h2>Population Percentages</h2>",
            "<p class=\"muted\">This ledger does not include biomass totals. Re-run inference with the current <code>--biomass-ledger</code> implementation to enable this plot.</p>",
        ])
    html_parts.append("<h2>Biomass Loss and Extinction Risk</h2>")
    if has_biomass_totals:
        html_parts.extend([
            "<p class=\"muted\">Species are ranked by percentage decline from their starting biomass. The top three declining species get detailed timelines below.</p>",
            _html_table(
                ["Rank", "Species", "Starting biomass", "Ending biomass", "End (% of start)", "Decline"],
                biomass_table_rows,
            ),
        ])
        for idx, loss_detail_path, gain_detail_path, top_causes, top_gain_sources in extinction_detail_paths:
            loss_rel = loss_detail_path.relative_to(outdir)
            gain_rel = gain_detail_path.relative_to(outdir)
            cause_rows = [
                [rank, cause, _fmt(value)]
                for rank, (cause, value) in enumerate(top_causes, start=1)
            ]
            gain_rows = [
                [rank, source, _fmt(value)]
                for rank, (source, value) in enumerate(top_gain_sources, start=1)
            ]
            html_parts.extend([
                f"<h2>{html.escape(_pretty_label(fg_ids[idx]))}: Decline Detail</h2>",
                "<div class=\"cards\">",
                f"<div class=\"card\"><div class=\"label\">Start</div><div class=\"value\">{_fmt(initial_biomass[idx])}</div></div>",
                f"<div class=\"card\"><div class=\"label\">End</div><div class=\"value\">{_fmt(final_biomass[idx])}</div></div>",
                f"<div class=\"card\"><div class=\"label\">End of Start</div><div class=\"value\">{float(end_pct[idx]):.2f}%</div></div>",
                "</div>",
                "<div class=\"figure\">",
                f"<img src=\"{html.escape(str(loss_rel))}\" alt=\"Decline detail for {html.escape(fg_ids[idx])}\">",
                "</div>",
                _html_table(["Rank", "Logged loss cause", "Total tonnes"], cause_rows),
                "<div class=\"figure\">",
                f"<img src=\"{html.escape(str(gain_rel))}\" alt=\"Gain detail for declining {html.escape(fg_ids[idx])}\">",
                "</div>",
                _html_table(["Rank", "Logged gain source", "Total tonnes"], gain_rows),
            ])
    else:
        html_parts.append(
            "<p class=\"muted\">This ledger does not include biomass totals. Re-run inference with the current <code>--biomass-ledger</code> implementation to enable this section.</p>"
        )
    html_parts.extend([
        "<h2>Action Decisions Over Time</h2>",
    ])
    if has_action_data:
        html_parts.append(
            "<p class=\"muted\">Percentages are biomass-weighted: a value of 40% means 40% of that species' biomass was assigned to that action class on that tick. Detailed actions include movement directions, rest, and prey-specific eat actions.</p>"
        )
        for dm_id, action_path, mean_categories in action_detail_paths:
            rel = action_path.relative_to(outdir)
            mean_rows = [
                [rank, _pretty_label(action_category_labels[idx]), f"{float(mean_categories[idx]):.2f}%"]
                for rank, idx in enumerate(np.argsort(mean_categories)[::-1], start=1)
            ]
            html_parts.extend([
                f"<h2>{html.escape(_pretty_label(dm_id))}: Decisions</h2>",
                "<div class=\"figure\">",
                f"<img src=\"{html.escape(str(rel))}\" alt=\"Action decisions for {html.escape(dm_id)}\">",
                "</div>",
                _html_table(["Rank", "Action category", "Mean over ticks"], mean_rows),
            ])
    else:
        html_parts.append(
            "<p class=\"muted\">This ledger does not include action percentages. Re-run inference with the current <code>--biomass-ledger</code> implementation to enable this section.</p>"
        )
    html_parts.extend([
        "<h2>Top Predation Flows</h2>",
        _html_table(["Rank", "Predator", "Prey", "Tonnes consumed", "Assimilated energy"], top_pred_rows),
    ])

    figure_sections = [
        ("Predation Matrix", "predation_heatmap"),
        ("Assimilation Matrix", "assimilation_heatmap"),
        ("Diet Composition", "diet_bars"),
        ("Loss Composition", "loss_bars"),
        ("Gain Composition", "gain_bars"),
        ("Predation Through Time", "predation_lines"),
        ("Gain Sources Through Time", "gain_lines"),
    ]
    for title, key in figure_sections:
        rel = paths[key].relative_to(outdir)
        html_parts.extend([
            f"<h2>{html.escape(title)}</h2>",
            "<div class=\"figure\">",
            f"<img src=\"{html.escape(str(rel))}\" alt=\"{html.escape(title)}\">",
            "</div>",
        ])

    html_parts.extend([
        "<h2>Files</h2>",
        "<p>The report directory also contains <code>species_flow_summary.csv</code> and the PNG files under <code>figures/</code>.</p>",
        "</main></body></html>",
    ])
    report_path.write_text("\n".join(html_parts), encoding="utf-8")
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a Mareld biomass ledger .npz as an HTML report."
    )
    parser.add_argument("ledger", help="Path to a ledger .npz created by inference.py --biomass-ledger.")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to <ledger-stem>_report beside the ledger.",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of top flows/sources to highlight.")
    return parser.parse_args()


def main():
    args = parse_args()
    ledger_path = Path(args.ledger)
    if not ledger_path.is_file():
        raise SystemExit(f"Ledger not found: {ledger_path}")
    outdir = Path(args.outdir) if args.outdir else ledger_path.with_suffix("").parent / f"{ledger_path.stem}_report"
    report_path = build_report(ledger_path, outdir, top_n=max(1, int(args.top)))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
