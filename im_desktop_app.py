#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IM Desktop Viewer (Offline, No Browser)
PySide6 desktop application to parse .im (XML) RF files and display 4 charts:
  • Gt(dB) @ f0 vs Pout(dBm) @ f0
  • AM/PM(offset) @ f0 vs Pout(dBm) @ f0
  • Drain Efficiency(%) @ f0 vs Pout(dBm) @ f0
  • Input Return Loss(dB) @ f0 vs Pout(dBm) @ f0

Features
- File > Open .im
- Single-curve mode OR S1/S3 overlay mode (auto-detect + manual selection)
- Optional Γ_source scaling for Pavs (if "Gamma Source" exists in the file)
- Export: Save CSV (metrics) and Save All Plots (PNGs)
- View > Diagnostics…  (lists which columns were detected per curve)

Build (Windows one-file EXE):
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name "IMDesktop" ^
  --collect-all PySide6 ^
  --collect-all matplotlib ^
  im_desktop_app.py
"""

import sys, math, xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

# Matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ------------------------- Core parsing and metrics -------------------------

@dataclass
class CurveRecord:
    dataset_name: str
    curve_name: str
    cols: Dict[str, List[Any]]
    meta: Dict[str, Dict[str, str]]
    rows: int

def _to_complex(val) -> complex:
    if isinstance(val, complex):
        return val
    if isinstance(val, tuple) and len(val) == 2:
        try:
            return complex(float(val[0]), float(val[1]))
        except Exception:
            return complex(np.nan, np.nan)
    if isinstance(val, (int, float)) and np.isfinite(val):
        return complex(val, 0.0)
    return complex(np.nan, np.nan)

def parse_im_file(path: Path) -> List[CurveRecord]:
    """Parse .im XML and return a list of CurveRecord objects."""
    tree = ET.parse(str(path))
    root = tree.getroot()
    out: List[CurveRecord] = []
    for dataset in root.findall(".//dataset"):
        dname = dataset.get("name", "")
        for curve in dataset.findall("./curve"):
            cname = curve.get("name", "")
            cols: Dict[str, List[Any]] = {}
            meta: Dict[str, Dict[str, str]] = {}
            max_len = 0
            for node in curve.findall(".//data"):
                cid = node.get("id") or node.get("name") or f"col_{len(cols)}"
                name = node.get("name") or cid
                unit = node.get("unit") or ""
                text = (node.text or "").strip()
                if text.startswith("[") and text.endswith("]"):
                    text = text[1:-1]
                items = text.split(",") if text else []
                vals: List[Any] = []
                for s in items:
                    s = s.strip()
                    if not s:
                        vals.append(np.nan); continue
                    if " " in s and not any(ch.isalpha() for ch in s):
                        parts = s.split()
                        if len(parts) == 2:
                            try:
                                vals.append((float(parts[0]), float(parts[1])))
                                continue
                            except Exception:
                                vals.append(np.nan); continue
                    try:
                        vals.append(float(s))
                    except Exception:
                        vals.append(np.nan)
                cols[cid] = vals
                meta[cid] = {"name": name, "unit": unit}
                max_len = max(max_len, len(vals))
            for k, v in list(cols.items()):
                if len(v) < max_len:
                    cols[k] = v + [np.nan] * (max_len - len(v))
            out.append(CurveRecord(dname, cname, cols, meta, max_len))
    return out

def _pick_col(meta: Dict[str, Dict[str, str]], prefix: str) -> Optional[str]:
    pl = prefix.lower()
    for cid, m in meta.items():
        nm = (m.get("name") or cid).lower()
        if nm.startswith(pl):
            return cid
    return None

def compute_metrics(record: CurveRecord, use_gamma_source: bool=False) -> pd.DataFrame:
    """Compute Pout[dBm], Gt[dB], AM/PM offset[deg], Drain Eff[%], Input RL[dB] for a curve.
    Robust to missing A2/B1: if a wave is missing/NaN, fall back to simple definitions.
    """
    cols, meta, rows = record.cols, record.meta, record.rows

    def arr_complex(cid):
        if cid is None: return np.full(rows, complex(np.nan, np.nan))
        raw = np.asarray(cols[cid], dtype=object)
        out = np.empty(rows, dtype=complex)
        for i, v in enumerate(raw):
            out[i] = _to_complex(v)
        return out

    def arr_float(cid):
        if cid is None: return np.full(rows, np.nan)
        raw = np.asarray(cols[cid], dtype=object)
        out = np.full(rows, np.nan)
        for i, v in enumerate(raw):
            if isinstance(v, (int, float)) and np.isfinite(v):
                out[i] = float(v)
        return out

    A1 = _pick_col(meta, "A1"); A2 = _pick_col(meta, "A2")
    B1 = _pick_col(meta, "B1"); B2 = _pick_col(meta, "B2")
    Pdc = next((cid for cid, m in meta.items() if (m.get("name","").lower().startswith("pdc"))), None)
    Gs  = next((cid for cid, m in meta.items() if "gamma source" in (m.get("name","").lower())), None)

    a1, a2, b1, b2 = arr_complex(A1), arr_complex(A2), arr_complex(B1), arr_complex(B2)
    pdc = arr_float(Pdc)
    gamma_s = arr_complex(Gs) if use_gamma_source else np.full(rows, complex(0.0, 0.0))

    # Detect missing/empty waves
    has_a2 = np.isfinite(np.real(a2)).any()
    has_b1 = np.isfinite(np.real(b1)).any()

    # Delivered output power: prefer net (|B2|^2 - |A2|^2), else simple (|B2|^2)
    if has_a2:
        pout_w = np.maximum(np.abs(b2)**2 - np.abs(a2)**2, 0.0)
    else:
        pout_w = np.abs(b2)**2  # fallback if A2 missing

    # Available source power (approx): |A1|^2, optionally scaled by (1 - |ΓS|^2)
    pavs_w = np.abs(a1)**2 * (1.0 - np.minimum(np.abs(gamma_s)**2, 0.999999))

    # Convert to dBm
    pout_dbm = 10.0 * np.log10(np.maximum(pout_w, 1e-12) / 1e-3)

    # Transducer gain (Gt)
    gt_db = 10.0 * np.log10(np.maximum(pout_w / np.where(pavs_w > 1e-18, pavs_w, np.nan), 1e-18))

    # AM/PM offset (relative phase, unwrapped)
    phase_rel = np.unwrap(np.angle(b2) - np.angle(a1))
    finite = np.isfinite(phase_rel) & np.isfinite(pout_dbm)
    ref = phase_rel[finite][0] if np.any(finite) else 0.0
    ampm_deg = (phase_rel - ref) * 180.0 / math.pi

    # Drain efficiency
    drain_eff = np.where(pdc > 0, (pout_w / pdc) * 100.0, np.nan)

    # Input return loss: only valid if A1 & B1 present; else NaN
    if has_b1:
        gamma_in = np.where(np.abs(a1) > 0, b1 / a1, complex(np.nan, np.nan))
        irl_db = -20.0 * np.log10(np.clip(np.abs(gamma_in), 1e-12, 1.0))
    else:
        irl_db = np.full(rows, np.nan)

    df = pd.DataFrame({
        "Pout [dBm] @ f0": pout_dbm,
        "Gt [dB] @ f0": gt_db,
        "AM/PM offset [deg] @ f0": ampm_deg,
        "Drain Efficiency [%] @ f0": drain_eff,
        "Input Return Loss [dB] @ f0": irl_db,
    })
    return df


# ------------------------- UI components -------------------------

class PlotGrid(QtWidgets.QWidget):
    """Four Matplotlib plots in a 2x2 grid."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(9, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax_gt = self.figure.add_subplot(221)
        self.ax_ampm = self.figure.add_subplot(222)
        self.ax_eff = self.figure.add_subplot(223)
        self.ax_irl = self.figure.add_subplot(224)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def clear(self):
        for ax in (self.ax_gt, self.ax_ampm, self.ax_eff, self.ax_irl):
            ax.clear()

    def plot_single(self, df: pd.DataFrame, title_suffix: str = ""):
        self.clear()
        x = df["Pout [dBm] @ f0"].values

        m = np.isfinite(x) & np.isfinite(df["Gt [dB] @ f0"].values)
        self.ax_gt.plot(x[m], df["Gt [dB] @ f0"].values[m], marker="o")
        self.ax_gt.set_xlabel("Pout [dBm] @ f0"); self.ax_gt.set_ylabel("Gt [dB] @ f0")
        self.ax_gt.set_title(f"Gt vs Pout {title_suffix}".strip())

        m2 = np.isfinite(x) & np.isfinite(df["AM/PM offset [deg] @ f0"].values)
        self.ax_ampm.plot(x[m2], df["AM/PM offset [deg] @ f0"].values[m2], marker="o")
        self.ax_ampm.set_xlabel("Pout [dBm] @ f0"); self.ax_ampm.set_ylabel("AM/PM offset [deg] @ f0")
        self.ax_ampm.set_title(f"AM/PM vs Pout {title_suffix}".strip())

        m3 = np.isfinite(x) & np.isfinite(df["Drain Efficiency [%] @ f0"].values)
        self.ax_eff.plot(x[m3], df["Drain Efficiency [%] @ f0"].values[m3], marker="o")
        self.ax_eff.set_xlabel("Pout [dBm] @ f0"); self.ax_eff.set_ylabel("Drain Efficiency [%] @ f0")
        self.ax_eff.set_title(f"Drain Efficiency vs Pout {title_suffix}".strip())

        m4 = np.isfinite(x) & np.isfinite(df["Input Return Loss [dB] @ f0"].values)
        self.ax_irl.plot(x[m4], df["Input Return Loss [dB] @ f0"].values[m4], marker="o")
        self.ax_irl.set_xlabel("Pout [dBm] @ f0"); self.ax_irl.set_ylabel("Input Return Loss [dB] @ f0")
        self.ax_irl.set_title(f"Input RL vs Pout {title_suffix}".strip())

        self.canvas.draw_idle()

    def plot_overlay(self, df1: pd.DataFrame, lbl1: str, df2: pd.DataFrame, lbl2: str, title_suffix: str = ""):
        self.clear()
        x1 = df1["Pout [dBm] @ f0"].values; x2 = df2["Pout [dBm] @ f0"].values

        m1 = np.isfinite(x1) & np.isfinite(df1["Gt [dB] @ f0"].values)
        m2 = np.isfinite(x2) & np.isfinite(df2["Gt [dB] @ f0"].values)
        self.ax_gt.plot(x1[m1], df1["Gt [dB] @ f0"].values[m1], marker="o", label=lbl1)
        self.ax_gt.plot(x2[m2], df2["Gt [dB] @ f0"].values[m2], marker="s", label=lbl2)
        self.ax_gt.set_xlabel("Pout [dBm] @ f0"); self.ax_gt.set_ylabel("Gt [dB] @ f0")
        self.ax_gt.set_title(f"Gt vs Pout {title_suffix}".strip()); self.ax_gt.legend()

        ma1 = np.isfinite(x1) & np.isfinite(df1["AM/PM offset [deg] @ f0"].values)
        ma2 = np.isfinite(x2) & np.isfinite(df2["AM/PM offset [deg] @ f0"].values)
        self.ax_ampm.plot(x1[ma1], df1["AM/PM offset [deg] @ f0"].values[ma1], marker="o", label=lbl1)
        self.ax_ampm.plot(x2[ma2], df2["AM/PM offset [deg] @ f0"].values[ma2], marker="s", label=lbl2)
        self.ax_ampm.set_xlabel("Pout [dBm] @ f0"); self.ax_ampm.set_ylabel("AM/PM offset [deg] @ f0")
        self.ax_ampm.set_title(f"AM/PM vs Pout {title_suffix}".strip()); self.ax_ampm.legend()

        me1 = np.isfinite(x1) & np.isfinite(df1["Drain Efficiency [%] @ f0"].values)
        me2 = np.isfinite(x2) & np.isfinite(df2["Drain Efficiency [%] @ f0"].values)
        self.ax_eff.plot(x1[me1], df1["Drain Efficiency [%] @ f0"].values[me1], marker="o", label=lbl1)
        self.ax_eff.plot(x2[me2], df2["Drain Efficiency [%] @ f0"].values[me2], marker="s", label=lbl2)
        self.ax_eff.set_xlabel("Pout [dBm] @ f0"); self.ax_eff.set_ylabel("Drain Efficiency [%] @ f0")
        self.ax_eff.set_title(f"Drain Efficiency vs Pout {title_suffix}".strip()); self.ax_eff.legend()

        mi1 = np.isfinite(x1) & np.isfinite(df1["Input Return Loss [dB] @ f0"].values)
        mi2 = np.isfinite(x2) & np.isfinite(df2["Input Return Loss [dB] @ f0"].values)
        self.ax_irl.plot(x1[mi1], df1["Input Return Loss [dB] @ f0"].values[mi1], marker="o", label=lbl1)
        self.ax_irl.plot(x2[mi2], df2["Input Return Loss [dB] @ f0"].values[mi2], marker="s", label=lbl2)
        self.ax_irl.set_xlabel("Pout [dBm] @ f0"); self.ax_irl.set_ylabel("Input Return Loss [dB] @ f0")
        self.ax_irl.set_title(f"Input RL vs Pout {title_suffix}".strip()); self.ax_irl.legend()

        self.canvas.draw_idle()

    def save_all(self, out_dir: Path, prefix: str = "plots"):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(out_dir / f"{prefix}__4up.png", dpi=150)


class ControlPanel(QtWidgets.QWidget):
    request_update = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.overlay_cb = QtWidgets.QCheckBox("Overlay S1 & S3 (if present)")
        self.gamma_cb = QtWidgets.QCheckBox("Use Gamma Source (scale Pavs by 1−|Γs|²)")
        self.curve_combo = QtWidgets.QComboBox()
        self.trace1_combo = QtWidgets.QComboBox()
        self.trace2_combo = QtWidgets.QComboBox()

        form = QtWidgets.QFormLayout()
        form.addRow(self.overlay_cb)
        form.addRow(self.gamma_cb)
        form.addRow("Single curve:", self.curve_combo)
        form.addRow("Trace 1 (overlay):", self.trace1_combo)
        form.addRow("Trace 2 (overlay):", self.trace2_combo)

        # Buttons
        self.export_csv_btn = QtWidgets.QPushButton("Export Metrics CSV")
        self.export_plots_btn = QtWidgets.QPushButton("Save All Plots (PNG)")
        form.addRow(self.export_csv_btn)
        form.addRow(self.export_plots_btn)

        self.setLayout(form)

        # Signals -> trigger update
        for w in (self.overlay_cb, self.gamma_cb, self.curve_combo, self.trace1_combo, self.trace2_combo):
            if isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(self.request_update.emit)
            elif isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(self.request_update.emit)

    def set_labels(self, labels: List[str]):
        self.curve_combo.blockSignals(True)
        self.trace1_combo.blockSignals(True)
        self.trace2_combo.blockSignals(True)
        self.curve_combo.clear(); self.trace1_combo.clear(); self.trace2_combo.clear()
        self.curve_combo.addItems(labels)
        self.trace1_combo.addItems(labels)
        self.trace2_combo.addItems(labels)
        self.curve_combo.blockSignals(False)
        self.trace1_combo.blockSignals(False)
        self.trace2_combo.blockSignals(False)

    def auto_pick_s1s3(self, labels: List[str]):
        i_s1 = next((i for i, s in enumerate(labels) if "s1" in s.lower() and "1-tone" in s.lower()), None)
        i_s3 = next((i for i, s in enumerate(labels) if "s3" in s.lower() and "1-tone" in s.lower()), None)
        if i_s1 is not None:
            self.trace1_combo.setCurrentIndex(i_s1)
        if i_s3 is not None:
            self.trace2_combo.setCurrentIndex(i_s3)

    def is_overlay(self) -> bool:
        return self.overlay_cb.isChecked()

    def use_gamma(self) -> bool:
        return self.gamma_cb.isChecked()

    def current_curve_index(self) -> int:
        return self.curve_combo.currentIndex()

    def trace_indices(self) -> Tuple[int, int]:
        return self.trace1_combo.currentIndex(), self.trace2_combo.currentIndex()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IM Desktop Viewer")
        self.resize(1200, 800)

        self.curves: List[CurveRecord] = []
        self.labels: List[str] = []

        # Widgets
        self.plot_grid = PlotGrid(self)
        self.controls = ControlPanel(self)

        # Dock for controls
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setWidget(self.controls)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        self.setCentralWidget(self.plot_grid)

        # Menu
        file_menu = self.menuBar().addMenu("&File")
        open_act = QtGui.QAction("Open .im...", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.on_open)
        file_menu.addAction(open_act)

        exit_act = QtGui.QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        view_menu = self.menuBar().addMenu("&View")
        diag_act = QtGui.QAction("Diagnostics…", self)
        diag_act.triggered.connect(self.show_diagnostics)
        view_menu.addAction(diag_act)

        # Export actions connect to control buttons
        self.controls.export_csv_btn.clicked.connect(self.export_csv)
        self.controls.export_plots_btn.clicked.connect(self.export_plots)
        self.controls.request_update.connect(self.update_plots)

        self.statusBar().showMessage("Open a .im file to begin.")

    # ----- File handling -----
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open .im File", "", "IM/XML Files (*.im *.xml);;All Files (*.*)")
        if not path:
            return
        try:
            self.curves = parse_im_file(Path(path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Parse Error", f"Failed to parse XML:\\n{e}")
            return

        self.labels = [f"{c.dataset_name} / {c.curve_name}" for c in self.curves]
        self.controls.set_labels(self.labels)
        self.controls.auto_pick_s1s3(self.labels)
        self.statusBar().showMessage(f"Loaded {len(self.curves)} curves from: {path}")

        # Initial plot
        self.update_plots()

    # ----- Diagnostics -----
    def show_diagnostics(self):
        if not self.curves:
            QtWidgets.QMessageBox.information(self, "Diagnostics", "No file loaded.")
            return
        msgs = []
        for i, rec in enumerate(self.curves):
            cols = set(rec.cols.keys())
            names = {cid: (rec.meta.get(cid, {}).get('name') or cid) for cid in cols}
            keys = []
            for k in ["A1","A2","B1","B2","Pdc","Gamma Source"]:
                if any((names[c].lower().startswith(k.lower())) for c in cols):
                    keys.append(k)
            msgs.append(f"[{i}] {rec.dataset_name} / {rec.curve_name}\\n  Found: " + (', '.join(keys) if keys else 'none'))
        QtWidgets.QMessageBox.information(self, "Diagnostics", "\\n\\n".join(msgs))

    # ----- Plotting -----
    def update_plots(self):
        if not self.curves:
            return
        overlay = self.controls.is_overlay()
        use_gamma = self.controls.use_gamma()

        if not overlay:
            idx = self.controls.current_curve_index()
            idx = max(0, min(idx, len(self.curves)-1))
            rec = self.curves[idx]
            df = compute_metrics(rec, use_gamma_source=use_gamma)
            title = f"({rec.curve_name})"
            self.plot_grid.plot_single(df, title_suffix=title)
            self._last_df = df
            self._last_df2 = None
            self._last_labels = (self.labels[idx], None)
        else:
            i1, i2 = self.controls.trace_indices()
            i1 = max(0, min(i1, len(self.curves)-1))
            i2 = max(0, min(i2, len(self.curves)-1))
            rec1, rec2 = self.curves[i1], self.curves[i2]
            df1 = compute_metrics(rec1, use_gamma_source=use_gamma)
            df2 = compute_metrics(rec2, use_gamma_source=use_gamma)
            title = f"({rec1.curve_name} vs {rec2.curve_name})"
            self.plot_grid.plot_overlay(df1, self.labels[i1], df2, self.labels[i2], title_suffix=title)
            self._last_df = df1
            self._last_df2 = df2
            self._last_labels = (self.labels[i1], self.labels[i2])

    # ----- Export -----
    def export_csv(self):
        if not hasattr(self, "_last_df") or self._last_df is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Metrics CSV", "metrics.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            if self._last_df2 is None:
                self._last_df.to_csv(path, index=False)
            else:
                # Combine with suffixes for overlay
                df_combined = pd.concat([
                    self._last_df.add_suffix(" (Trace1)"),
                    self._last_df2.add_suffix(" (Trace2)")
                ], axis=1)
                df_combined.to_csv(path, index=False)
            self.statusBar().showMessage(f"Saved CSV: {path}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save CSV:\\n{e}")

    def export_plots(self):
        if not hasattr(self, "plot_grid"):
            return
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Folder to Save Plots")
        if not dir_path:
            return
        try:
            prefix = "S1S3" if self.controls.is_overlay() else "Single"
            self.plot_grid.save_all(Path(dir_path), prefix=prefix)
            self.statusBar().showMessage(f"Saved plots to: {dir_path}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save plots:\\n{e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("IM Desktop Viewer")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
