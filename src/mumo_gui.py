"""Interactive desktop calculator for the Mumo framework."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Dict, List

from mumo_framework import MumoAnalysis, analyze_dataset


_SAMPLE_DATA = "10, 12.5, 14, 15.5, 18, 21, 25, 28.5, 32, 36"


def _parse_dataset(text: str) -> List[float]:
    """Parse comma- or whitespace-separated finite numbers."""
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("Enter at least three data values.")

    values = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError("Could not read {!r} as a number.".format(token)) from exc
        if not math.isfinite(value):
            raise ValueError("All data values must be finite numbers.")
        values.append(value)
    if len(values) < 3:
        raise ValueError("Enter at least three data values.")
    return values


def _calculate_results(
    data: List[float],
    k: int,
    alpha: float,
    beta: float,
    gamma: float,
    compare_standard: bool = False,
) -> MumoAnalysis:
    """Calculate the displayed metrics for the current inputs."""
    return analyze_dataset(
        data, k, alpha, beta, gamma, compare_standard=compare_standard
    )


class MumoCalculator:
    """Tkinter interface with live dataset and parameter updates."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mumo Framework | Interactive Calculator")
        self.root.minsize(760, 650)
        self.root.geometry("940x800")
        self.root.configure(background="#f3f6fb")
        self._pending_update: str | None = None

        self.k = tk.IntVar(value=2)
        self.alpha = tk.DoubleVar(value=0.6)
        self.beta = tk.DoubleVar(value=0.5)
        self.gamma = tk.DoubleVar(value=0.5)
        self.compare_standard = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Edit the data or adjust a parameter to recalculate.")
        self.count = tk.StringVar(value="0 values")
        self._metric_values: Dict[str, tk.StringVar] = {}

        self._style()
        self._build()
        self._set_sample()

    def _style(self) -> None:
        style = ttk.Style(self.root)
        style.configure("TFrame", background="#f3f6fb")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure(
            "Title.TLabel",
            background="#f3f6fb",
            foreground="#14213d",
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#f3f6fb",
            foreground="#65748b",
            font=("Segoe UI", 10),
        )
        style.configure(
            "PanelTitle.TLabel",
            background="#ffffff",
            foreground="#14213d",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "MetricName.TLabel",
            background="#ffffff",
            foreground="#65748b",
            font=("Segoe UI", 9),
        )
        style.configure(
            "MetricValue.TLabel",
            background="#ffffff",
            foreground="#14213d",
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "HighlightValue.TLabel",
            background="#ffffff",
            foreground="#3157d5",
            font=("Segoe UI", 18, "bold"),
        )
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 6))
        style.configure("TScale", background="#ffffff")

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=(24, 20))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Mumo Interactive Calculator", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            outer,
            text="Explore how your dataset and model parameters shape each index.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 18))

        columns = ttk.Frame(outer)
        columns.pack(fill="x")
        columns.columnconfigure(0, weight=3)
        columns.columnconfigure(1, weight=2)

        data_panel = ttk.Frame(columns, style="Panel.TFrame", padding=16)
        data_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(data_panel, text="Dataset", style="PanelTitle.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            data_panel,
            text="Paste or type numbers, separated by commas, spaces, or new lines.",
            style="MetricName.TLabel",
        ).pack(anchor="w", pady=(3, 10))
        text_frame = ttk.Frame(data_panel, style="Panel.TFrame")
        text_frame.pack(fill="both", expand=True)
        self.dataset = tk.Text(
            text_frame,
            height=5,
            wrap="word",
            undo=True,
            font=("Consolas", 10),
            relief="solid",
            borderwidth=1,
            padx=9,
            pady=8,
        )
        scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.dataset.yview
        )
        self.dataset.configure(yscrollcommand=scrollbar.set)
        self.dataset.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.dataset.bind("<<Modified>>", self._on_dataset_change)

        data_actions = ttk.Frame(data_panel, style="Panel.TFrame")
        data_actions.pack(fill="x", pady=(9, 0))
        ttk.Button(data_actions, text="Load sample", command=self._set_sample).pack(
            side="left"
        )
        ttk.Button(data_actions, text="Clear", command=self._clear_dataset).pack(
            side="left", padx=(7, 0)
        )
        ttk.Label(data_actions, textvariable=self.count, style="MetricName.TLabel").pack(
            side="right", pady=6
        )

        parameters = ttk.Frame(columns, style="Panel.TFrame", padding=16)
        parameters.grid(row=0, column=1, sticky="nsew")
        ttk.Label(parameters, text="Parameters", style="PanelTitle.TLabel").pack(
            anchor="w", pady=(0, 5)
        )
        self._add_parameter(
            parameters, "Core / boundary size (k)", self.k, 1, 4, "{:d}",
            integer=True,
        )
        self._add_parameter(parameters, "Alpha", self.alpha, 0, 1, "{:.2f}")
        self._add_parameter(parameters, "Beta", self.beta, 0, 1, "{:.2f}")
        self._add_parameter(parameters, "Gamma", self.gamma, 0, 1, "{:.2f}")
        ttk.Checkbutton(
            parameters,
            text="Compare standard metrics",
            variable=self.compare_standard,
            command=self._on_comparison_change,
        ).pack(anchor="w", pady=(8, 0))

        results_panel = ttk.Frame(outer, style="Panel.TFrame", padding=16)
        results_panel.pack(fill="both", expand=True, pady=(16, 0))
        ttk.Label(results_panel, text="Calculated indices", style="PanelTitle.TLabel").pack(
            anchor="w", pady=(0, 12)
        )
        metrics = ttk.Frame(results_panel, style="Panel.TFrame")
        metrics.pack(fill="both", expand=True)
        metric_names = [
            "GMA-k", "MBTS", "P-MPU", "P-MPD", "P-GMA", "MBV", "P-MBTS",
            "P-GMCSI", "MAEI", "MCHI", "Standard Mean", "Standard Median",
            "Standard Deviation",
        ]
        for column in range(3):
            metrics.columnconfigure(column, weight=1)
        for index, name in enumerate(metric_names):
            row, column = divmod(index, 3)
            card = ttk.Frame(metrics, style="Panel.TFrame", padding=(8, 7))
            card.grid(row=row, column=column, sticky="nsew", padx=4, pady=3)
            ttk.Label(card, text=name, style="MetricName.TLabel").pack(anchor="w")
            value = tk.StringVar(value="—")
            self._metric_values[name] = value
            value_style = (
                "HighlightValue.TLabel" if name in ("MAEI", "MCHI") else "MetricValue.TLabel"
            )
            ttk.Label(card, textvariable=value, style=value_style).pack(anchor="w")

        self.status_label = ttk.Label(
            outer,
            textvariable=self.status,
            style="Subtitle.TLabel",
            wraplength=880,
        )
        self.status_label.pack(anchor="w", pady=(11, 0))

    def _add_parameter(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        minimum: int,
        maximum: int,
        number_format: str,
        integer: bool = False,
    ) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=5)
        heading = ttk.Frame(row, style="Panel.TFrame")
        heading.pack(fill="x")
        ttk.Label(heading, text=label, style="MetricName.TLabel").pack(side="left")
        value_label = ttk.Label(heading, style="PanelTitle.TLabel")
        value_label.pack(side="right")

        def update_label(*_args: object) -> None:
            value_label.configure(text=number_format.format(variable.get()))

        variable.trace_add("write", update_label)
        update_label()
        if integer:
            scale = tk.Scale(
                row,
                from_=minimum,
                to=maximum,
                resolution=1,
                showvalue=False,
                variable=variable,
                command=self._on_parameter_change,
                orient="horizontal",
                highlightthickness=0,
                borderwidth=0,
                background="#ffffff",
                troughcolor="#e8edf5",
                activebackground="#3157d5",
            )
        else:
            scale = ttk.Scale(
                row,
                from_=minimum,
                to=maximum,
                variable=variable,
                command=self._on_parameter_change,
            )
        scale.pack(fill="x", pady=(2, 0))
        if integer:
            self.k_scale = scale

    def _on_dataset_change(self, _event: tk.Event) -> None:
        if not self.dataset.edit_modified():
            return
        self.dataset.edit_modified(False)
        if self._pending_update is not None:
            self.root.after_cancel(self._pending_update)
        self._pending_update = self.root.after(250, self._recalculate)

    def _on_parameter_change(self, _value: str) -> None:
        self._recalculate()

    def _on_comparison_change(self) -> None:
        self._recalculate()

    def _set_sample(self) -> None:
        self.dataset.delete("1.0", "end")
        self.dataset.insert("1.0", _SAMPLE_DATA)
        self.dataset.edit_modified(False)
        self._recalculate()

    def _clear_dataset(self) -> None:
        self.dataset.delete("1.0", "end")
        self.dataset.edit_modified(False)
        self._recalculate()

    def _recalculate(self) -> None:
        self._pending_update = None
        try:
            data = _parse_dataset(self.dataset.get("1.0", "end-1c"))
        except ValueError as exc:
            self.count.set("— values")
            self.status.set(str(exc))
            self.status_label.configure(foreground="#b42318")
            for value in self._metric_values.values():
                value.set("—")
            return

        maximum_k = (len(data) - 1) // 2
        self.k_scale.configure(from_=1, to=maximum_k)
        if self.k.get() < 1 or self.k.get() > maximum_k:
            self.k.set(min(maximum_k, max(1, self.k.get())))
        self.count.set("{} values".format(len(data)))

        try:
            results = _calculate_results(
                data,
                self.k.get(),
                self.alpha.get(),
                self.beta.get(),
                self.gamma.get(),
                self.compare_standard.get(),
            )
        except (TypeError, ValueError) as exc:
            self.status.set("Could not calculate indices: {}".format(exc))
            self.status_label.configure(foreground="#b42318")
            for value in self._metric_values.values():
                value.set("—")
            return

        for name in self._metric_values:
            result = results.metrics.get(name)
            if results.standard_metrics is not None:
                standard_name = {
                    "Standard Mean": "mean",
                    "Standard Median": "median",
                    "Standard Deviation": "standard_deviation",
                }.get(name)
                if standard_name is not None:
                    result = results.standard_metrics[standard_name]
            self._metric_values[name].set(
                "—" if result is None else "{:.6g}".format(result)
            )
        status = "Boundary route: {}. P-MBTS uses the mean after removing the lowest and highest k values.".format(
            results.routing
        )
        if results.blind_spot_alerts:
            status += "\n" + "\n".join(results.blind_spot_alerts)
        self.status.set(status)
        self.status_label.configure(
            foreground="#b42318" if results.blind_spot_alerts else "#65748b"
        )


def main() -> int:
    """Launch the interactive desktop calculator."""
    root = tk.Tk()
    MumoCalculator(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
