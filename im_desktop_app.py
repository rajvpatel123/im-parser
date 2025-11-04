
from __future__ import annotations

import sys, traceback, re, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QCheckBox, QGroupBox, QScrollArea, QRadioButton, QButtonGroup,
    QTabWidget, QComboBox, QTableWidget, QTableWidgetItem, QLineEdit
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

LOGFILE = Path(__file__).with_suffix(".log")
SESSIONFILE = Path(__file__).with_suffix(".session.json")

def log_ex(ex: BaseException):
    try:
        LOGFILE.write_text("".join(traceback.format_exception(ex)), encoding="utf-8")
    except Exception:
        pass

PAIR_RE = re.compile(r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$')
FREQ_NAME_RE = re.compile(r'(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)', re.IGNORECASE)

def human_hz(hz: float):
    if hz is None or not np.isfinite(hz): return 'Freq ?'
    if hz >= 1e9: return f'{hz/1e9:.3g} GHz'
    if hz >= 1e6: return f'{hz/1e6:.3g} MHz'
    if hz >= 1e3: return f'{hz/1e3:.3g} kHz'
    return f'{hz:.3g} Hz'

class CurveRecord:
    def __init__(self, dataset_name: str, curve_name: str, cols: dict, meta: dict, rows: int,
                 freq_hz: Optional[float], freq_label: str):
        self.dataset_name = dataset_name
        self.curve_name = curve_name
        self.cols = cols
        self.meta = meta
        self.rows = rows
        self.freq_hz = freq_hz
        self.freq_label = freq_label

def to_complex_any(val) -> complex:
    if isinstance(val, complex): return val
    if isinstance(val, (tuple, list, np.ndarray)) and len(val) == 2:
        try: return complex(float(val[0]), float(val[1]))
        except Exception: return complex(np.nan, np.nan)
    if isinstance(val, (int, float)) and np.isfinite(val): return complex(val, 0.0)
    return complex(np.nan, np.nan)

def arr_complex(cols, n, cid):
    if cid is None: return np.full(n, complex(np.nan, np.nan), dtype=complex)
    raw = np.asarray(cols[cid], dtype=object)
    out = np.empty(n, dtype=complex)
    for i, v in enumerate(raw): out[i] = to_complex_any(v)
    return out

def arr_float(cols, n, cid):
    if cid is None: return np.full(n, np.nan)
    raw = np.asarray(cols[cid], dtype=object)
    out = np.full(n, np.nan)
    for i, v in enumerate(raw):
        if isinstance(v, (int,float)) and np.isfinite(v): out[i] = float(v)
    return out

def pick_col(meta, prefix):
    pl = prefix.lower()
    for cid, m in meta.items():
        nm = (m.get("name") or cid).lower()
        if nm.startswith(pl):
            return cid
    return None

class ExistingXMLParser:
    def name(self) -> str: return "ExistingXMLParser"
    def can_parse(self, path: Path) -> float:
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(str(path)).getroot()
            if root.findall(".//dataset") and root.findall(".//curve") and root.findall(".//data"):
                return 0.9
        except Exception:
            return 0.0
        return 0.6

    def parse(self, path: Path) -> List[CurveRecord]:
        import xml.etree.ElementTree as ET
        root = ET.parse(str(path)).getroot()
        out: List[CurveRecord] = []
        for ds in root.findall(".//dataset"):
            dname = ds.get("name", "")
            for cv in ds.findall("./curve"):
                cname = cv.get("name", "")
                cols, meta, n = self._parse_curve(cv)
                f_hz, f_label = self._detect_frequency(dname, cname, cols, meta, n)
                out.append(CurveRecord(dname, cname, cols, meta, n, f_hz, f_label))
        return out

    def _parse_curve(self, curve):
        cols, meta, n = {}, {}, 0
        for data in curve.findall(".//data"):
            cid = data.get("id") or data.get("name") or f"col_{len(cols)}"
            name = data.get("name") or cid
            unit = data.get("unit") or ""
            text = (data.text or "").strip() if data.text else ""
            if text.startswith("[") and text.endswith("]"): text = text[1:-1]
            items = [s.strip() for s in text.split(",")] if text else []
            vals = []
            for s in items:
                m = PAIR_RE.match(s)
                if m:
                    try:
                        vals.append((float(m.group(1)), float(m.group(2)))); continue
                    except Exception: pass
                try: vals.append(float(s))
                except Exception: vals.append(np.nan)
            cols[cid] = vals
            meta[cid] = {"name": name, "unit": unit}
            n = max(n, len(vals))
        for k, v in list(cols.items()):
            if len(v) < n: cols[k] = v + [np.nan] * (n - len(v))
        return cols, meta, n

    def _detect_frequency(self, dname: str, cname: str, cols: dict, meta: dict, n: int):
        freq_cid = None
        for key in meta.keys():
            nm = (meta[key].get("name") or key).lower()
            if nm.startswith("f0") or nm.startswith("freq") or nm.startswith("frequency"):
                freq_cid = key; break
        if freq_cid:
            raw = np.asarray(cols[freq_cid], dtype=object)
            nums = [float(v) for v in raw if isinstance(v,(int,float)) and np.isfinite(v)]
            if nums:
                med = float(np.median(nums))
                return med, human_hz(med)
        text = f"{dname} {cname}"
        m = FREQ_NAME_RE.search(text)
        if m:
            val = float(m.group("val")); unit = m.group("unit").lower()
            mult = {"ghz":1e9,"mhz":1e6,"khz":1e3,"hz":1.0}[unit]
            hz = val*mult
            return hz, f"{val:g} {unit.upper()}"
        return None, "Freq ?"

class NewVariantParser:
    def name(self) -> str: return "NewVariantParser"
    def can_parse(self, path: Path) -> float:
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(str(path)).getroot()
            if root.tag.lower() == "datasource" and root.findall(".//curve[@id='IqPowMod']"):
                return 0.95
        except Exception:
            pass
        return 0.0

    def parse(self, path: Path) -> List[CurveRecord]:
        import xml.etree.ElementTree as ET
        root = ET.parse(str(path)).getroot()
        if root.tag.lower() == "datasource":
            records: List[CurveRecord] = []
            for c in root.findall(".//curve[@id='IqPowMod']"):
                dname = c.get("id") or "IqPowMod"
                cname = c.get("name") or (c.get("level") or "")
                cols: Dict[str, list] = {}
                meta: Dict[str, dict] = {}
                nmax = 0
                for d in c.findall(".//data"):
                    did = d.get("id") or d.get("data") or f"col_{len(cols)}"
                    txt = (d.text or "").strip()
                    if txt.startswith("[") and txt.endswith("]"): txt = txt[1:-1]
                    vals = []
                    for token in txt.split(",") if txt else []:
                        token = token.strip()
                        try: vals.append(float(token))
                        except Exception: vals.append(np.nan)
                    cols[did] = vals
                    meta[did] = {"name": d.get("name") or did, "unit": d.get("unit") or ""}
                    nmax = max(nmax, len(vals))
                for k,v in list(cols.items()):
                    if len(v) < nmax: cols[k] = v + [np.nan]*(nmax-len(v))
                f_hz = None; f_label = "Freq ?"
                m = re.search(r"(\d+(?:\.\d+)?)\s*(GHz|MHz|kHz|Hz)", f"{dname} {cname}", flags=re.IGNORECASE)
                if m:
                    val = float(m.group(1)); unit = m.group(2).lower()
                    mult = {"ghz":1e9, "mhz":1e6, "khz":1e3, "hz":1.0}[unit]
                    f_hz = val*mult; f_label = f"{val:g} {unit.upper()}"
                records.append(CurveRecord(dname, cname, cols, meta, nmax, f_hz, f_label))
            if records:
                return records
        raise RuntimeError("NewVariantParser: could not parse this file; adjust mappings for the new format.")

PARSERS = [ExistingXMLParser(), NewVariantParser()]

def pick_parser(path: Path):
    scores = []
    for p in PARSERS:
        try:
            scores.append((p, p.can_parse(path)))
        except Exception:
            scores.append((p, 0.0))
    best = max(scores, key=lambda t: t[1])
    if best[1] <= 0.0:
        raise RuntimeError("No parser recognized this .im file")
    return best[0]

def compute_metrics(record: CurveRecord, use_gamma_source: bool = False, ignore_a2: bool = False) -> pd.DataFrame:
    A1 = pick_col(record.meta, "A1"); A2 = pick_col(record.meta, "A2")
    B1 = pick_col(record.meta, "B1"); B2 = pick_col(record.meta, "B2")
    Pdc_cid = next((cid for cid, m in record.meta.items() if (m.get("name","").lower().startswith("pdc"))), None)
    Gs  = next((cid for cid, m in record.meta.items() if "gamma source" in (m.get("name","").lower())), None)

    a1 = arr_complex(record.cols, record.rows, A1)
    a2 = arr_complex(record.cols, record.rows, A2)
    b1 = arr_complex(record.cols, record.rows, B1)
    b2 = arr_complex(record.cols, record.rows, B2)
    pdc = arr_float(record.cols, record.rows, Pdc_cid)
    gamma_s = arr_complex(record.cols, record.rows, Gs) if use_gamma_source else np.full(record.rows, complex(0.0,0.0))

    def find_col_by_name(substrs):
        for cid, m in record.meta.items():
            nm = (m.get("name") or cid).lower()
            if all(s in nm for s in substrs):
                return cid
        return None
    acpr_lower_cid = find_col_by_name(["acpr","lower"]) or find_col_by_name(["aclr","lower"])
    acpr_upper_cid = find_col_by_name(["acpr","upper"]) or find_col_by_name(["aclr","upper"])
    acpr_lower = arr_float(record.cols, record.rows, acpr_lower_cid)
    acpr_upper = arr_float(record.cols, record.rows, acpr_upper_cid)

    have_waves = np.isfinite(np.real(b2)).any() or np.isfinite(np.real(a1)).any()

    if have_waves:
        has_a2 = np.isfinite(np.real(a2)).any()
        if ignore_a2 or not has_a2:
            b2 = b2/np.sqrt(2)
            pout_w = np.abs(b2*b2)
        else:
            pout_w = np.maximum(np.abs(b2)**2 - np.abs(a2)**2, 0.0)
        pavs_w = np.abs(a1)**2 * (1.0 - np.minimum(np.abs(gamma_s)**2, 0.999999))
        gt_db = 10*np.log10(np.maximum(pout_w / np.where(pavs_w > 1e-18, pavs_w, np.nan), 1e-18))
        phase_rel = np.unwrap(np.angle(b2) - np.angle(a1))
        finite = np.isfinite(phase_rel)
        ref = phase_rel[finite][0] if np.any(finite) else 0.0
        ampm_deg = (phase_rel - ref) * 180/np.pi
        has_b1 = np.isfinite(np.real(b1)).any()
        if has_b1:
            gamma_in = np.full(record.rows, complex(np.nan, np.nan), dtype=complex)
            denom_ok = (np.abs(a1) > 0) & np.isfinite(a1) & np.isfinite(b1)
            np.divide(b1, a1, out=gamma_in, where=denom_ok)
            irl_db = -20*np.log10(np.clip(np.abs(gamma_in), 1e-12, 1.0))
        else:
            irl_db = np.full(record.rows, np.nan)
    else:
        Pout_cid = next((cid for cid, m in record.meta.items() if (m.get("name","Pout").lower().startswith("pout")) or cid.lower()=="pout"), None)
        PinAvail_cid = next((cid for cid, m in record.meta.items() if (m.get("name","PinAvail").lower().startswith("pin avail")) or cid.lower()=="pinavail"), None)
        pout_w = arr_float(record.cols, record.rows, Pout_cid)
        pin_av_w = arr_float(record.cols, record.rows, PinAvail_cid)
        gt_db = 10*np.log10(np.maximum(pout_w / np.where(pin_av_w > 1e-18, pin_av_w, np.nan), 1e-18))
        ampm_deg = np.full(record.rows, np.nan)
        irl_db = np.full(record.rows, np.nan)

    pout_dbm = 10*np.log10(np.maximum(pout_w, 1e-12)/1e-3)
    drain_eff = np.where(pdc > 0, (pout_w / pdc) * 100.0, np.nan)

    df = pd.DataFrame({
        "Pout [dBm] @ f0": pout_dbm,
        "Gt [dB] @ f0": gt_db,
        "AM/PM offset [deg] @ f0": ampm_deg,
        "Drain Efficiency [%] @ f0": drain_eff,
        "Input Return Loss [dB] @ f0": irl_db,
        "ACPR Lower [dBc] @ f0": acpr_lower,
        "ACPR Upper [dBc] @ f0": acpr_upper,
    })
    return df

def record_to_dataframe_with_metrics(record: CurveRecord, use_gamma_source: bool = False, ignore_a2: bool = False) -> pd.DataFrame:
    n = record.rows
    cols_out = {}
    for cid, vals in record.cols.items():
        name = (record.meta.get(cid, {}).get("name") or cid)
        if len(vals) and isinstance(vals[0], (tuple, list, complex)):
            re_arr = np.full(n, np.nan, dtype=float)
            im_arr = np.full(n, np.nan, dtype=float)
            for i, v in enumerate(vals):
                if isinstance(v, complex):
                    re_arr[i] = np.real(v); im_arr[i] = np.imag(v)
                elif isinstance(v, (tuple, list)) and len(v) == 2:
                    try:
                        re_arr[i] = float(v[0]); im_arr[i] = float(v[1])
                    except Exception:
                        pass
                elif isinstance(v, (int,float)) and np.isfinite(v):
                    re_arr[i] = float(v); im_arr[i] = 0.0
            cols_out[f"{name}_re"] = re_arr
            cols_out[f"{name}_im"] = im_arr
        else:
            arr = np.full(n, np.nan, dtype=float)
            for i, v in enumerate(vals):
                if isinstance(v, (int,float)) and np.isfinite(v):
                    arr[i] = float(v)
            cols_out[name] = arr

    m = compute_metrics(record, use_gamma_source=use_gamma_source, ignore_a2=ignore_a2)
    for col in m.columns:
        cols_out[col] = m[col].values

    df = pd.DataFrame(cols_out)
    return df

def _extract_spectrum_pairs(record: CurveRecord):
    if "SpectrumFreq" not in record.cols or "SpectrumRawPower" not in record.cols:
        return []
    freqs = record.cols["SpectrumFreq"]
    powers = record.cols["SpectrumRawPower"]
    if isinstance(freqs, list) and freqs and isinstance(freqs[0], (int,float)) and isinstance(powers, list) and powers and isinstance(powers[0], (int,float)):
        return [(np.array(freqs, dtype=float), np.array(powers, dtype=float))]
    return []

# simple palette & line styles
LINESTYLES = ["-", "--", ":", "-."]

def file_color(idx: int):
    base = matplotlib.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    return base[idx % len(base)]

class PlotGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        gs = self.fig.add_gridspec(2, 2)
        self.ax_gain = self.fig.add_subplot(gs[0,0])
        self.ax_ampm_or_acpr = self.fig.add_subplot(gs[0,1])
        self.ax_eff  = self.fig.add_subplot(gs[1,0])
        self.ax_irl  = self.fig.add_subplot(gs[1,1])

        # paddings & toggles
        self._h_pad = 2.0
        self._w_pad = 1.0
        self._show_grid = False
        self.show_legends = True
        # legend placement: outside/right, below, or inside
        self.legend_mode = 'outside'  # 'outside' | 'below' | 'inside'
        self.legend_width = 0.22  # fraction reserved on right when outside
        self.legend_fs = 8  # font size
        self.legend_ncol = 2  # for 'below' mode
        self.compact_labels = True  # shorten labels (forced ON)

        # axis toggles
        self.show_gain = True
        self.show_ampm_or_acpr = True
        self.show_eff = True
        self.show_irl_or_spec = True

        # log scales per axis
        self.xscale_gain = "linear"; self.yscale_gain = "linear"
        self.xscale_ampm = "linear"; self.yscale_ampm = "linear"
        self.xscale_eff  = "linear"; self.yscale_eff  = "linear"
        self.xscale_irl  = "linear"; self.yscale_irl  = "linear"

        # color mapping per frequency
        self._freq_color = {}
        self._color_cycle = matplotlib.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])

        # cursor annotation
        self._cursor_text = None
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        # axis ranges (None = autoscale)
        self.ranges = {
            "gain":  {"x": [None, None], "y": [None, None]},
            "ampm":  {"x": [None, None], "y": [None, None]},
            "eff":   {"x": [None, None], "y": [None, None]},
            "irl":   {"x": [None, None], "y": [None, None]},
        }

    # setters
    def set_show_legends(self, show: bool): self.show_legends = bool(show)
    def set_hpad(self, v: float): self._h_pad = max(0.0, min(8.0, float(v)))
    def set_wpad(self, v: float): self._w_pad = max(0.0, min(8.0, float(v)))
    def set_show_grid(self, show: bool): self._show_grid = bool(show)
    def set_axis_toggle(self, gain:bool, ampm:bool, eff:bool, irl:bool):
        self.show_gain, self.show_ampm_or_acpr, self.show_eff, self.show_irl_or_spec = gain, ampm, eff, irl
    def set_axis_scales(self, xg, yg, xa, ya, xe, ye, xi, yi):
        self.xscale_gain, self.yscale_gain = xg, yg
        self.xscale_ampm, self.yscale_ampm = xa, ya
        self.xscale_eff,  self.yscale_eff  = xe, ye
        self.xscale_irl,  self.yscale_irl  = xi, yi
    def set_axis_ranges(self, key: str, xr: Tuple[Optional[float],Optional[float]], yr: Tuple[Optional[float],Optional[float]]):
        if key in self.ranges:
            self.ranges[key]["x"] = [xr[0], xr[1]]
            self.ranges[key]["y"] = [yr[0], yr[1]]


    def _short_label(self, label: str) -> str:
        if not self.compact_labels:
            return label
        # prefer part after bullet '•' if present, else tail after last slash
        if '•' in label:
            tail = label.split('•',1)[1].strip()
        else:
            tail = label.split('/')[-1]
        # limit length
        return (tail[:28] + '…') if len(tail) > 30 else tail


    def _color_for_freq(self, freq_label: str):
        key = str(freq_label or 'Freq ?')
        if key not in self._freq_color:
            idx = len(self._freq_color) % len(self._color_cycle)
            self._freq_color[key] = self._color_cycle[idx]
        return self._freq_color[key]

    def _apply_legend(self, ax):
        if not self.show_legends: return
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        # apply compact labels
        labels = [self._short_label(l) for l in labels]
        mode = self.legend_mode
        # If too many labels for outside, auto-fallback to below for this axes
        if mode == 'outside' and len(labels) > 8:
            mode = 'below'
        if mode == 'outside':
            ax.legend(handles, labels, loc='upper left',
                      bbox_to_anchor=(1.0 + self.legend_width*0.1, 1.0),
                      borderaxespad=0.0, fontsize=self.legend_fs, frameon=False)
        elif mode == 'below':
            # estimate rows = ceil(len(labels)/ncol) to scale bottom margin in draw()
            ncol = max(1, min(self.legend_ncol, len(labels)))
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      ncol=ncol, fontsize=self.legend_fs, frameon=False)
        else:
            # inside, top-right
            ax.legend(handles, labels, loc='upper right', fontsize=self.legend_fs, frameon=False)

    def clear_axes(self):
        for ax in [self.ax_gain, self.ax_ampm_or_acpr, self.ax_eff, self.ax_irl]:
            ax.cla()

    def apply_grid(self):
        for ax in [self.ax_gain, self.ax_ampm_or_acpr, self.ax_eff, self.ax_irl]:
            ax.grid(self._show_grid, alpha=0.3, linestyle='--')

    def draw(self):
        # detect if any axes will use 'below' legends
        any_below = False
        if self.show_legends:
            for ax in [self.ax_gain, self.ax_ampm_or_acpr, self.ax_eff, self.ax_irl]:
                h, l = ax.get_legend_handles_labels()
                if h:
                    labs = [self._short_label(x) for x in l]
                    mode = self.legend_mode
                    if mode == 'outside' and len(labs) > 8:
                        mode = 'below'
                    if mode == 'below':
                        any_below = True
                        break
        if self.show_legends and self.legend_mode == 'outside' and not any_below:
            right_margin = max(0.5, 1.0 - self.legend_width)
            bottom_margin = 0.0
        else:
            right_margin = 0.98
            # scale bottom for potential below legends
            bottom_margin = 0.12 if (self.show_legends and (self.legend_mode == 'below' or any_below)) else 0.0
        self.fig.tight_layout(rect=[0.0, bottom_margin, right_margin, 1.0], h_pad=self._h_pad, w_pad=self._w_pad)
        self.canvas.draw_idle()

    def _on_motion(self, event):
        if not event.inaxes: 
            if self._cursor_text is not None:
                self._cursor_text.set_visible(False)
                self.canvas.draw_idle()
            return
        ax = event.inaxes
        if self._cursor_text is None:
            self._cursor_text = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                            bbox=dict(boxstyle="round", fc="w", alpha=0.6), fontsize=8)
        self._cursor_text.set_text(f"x={event.xdata:.3g}, y={event.ydata:.3g}")
        self._cursor_text.xy = (event.xdata, event.ydata)
        self._cursor_text.set_visible(True)
        self.canvas.draw_idle()

    def reset_view(self):
        for ax in [self.ax_gain, self.ax_ampm_or_acpr, self.ax_eff, self.ax_irl]:
            try:
                ax.relim(); ax.autoscale()
            except Exception:
                pass
        self.draw()

    def save_figure(self, parent=None):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getSaveFileName(parent, "Save Figure", "plots.png", "PNG (*.png);;PDF (*.pdf)")
        if not path: return
        try:
            self.fig.savefig(path, bbox_inches='tight', dpi=150)
        except Exception as e:
            QMessageBox.critical(parent, "Save failed", str(e))

    def _apply_axis_ranges(self, ax, key: str):
        xr = self.ranges[key]["x"]
        yr = self.ranges[key]["y"]
        try:
            ax.set_xlim(left=xr[0], right=xr[1])
        except Exception: pass
        try:
            ax.set_ylim(bottom=yr[0], top=yr[1])
        except Exception: pass

    def overlay_frequency_curves(self, mapping: Dict[str, Tuple[int, List[CurveRecord]]], selected_labels: List[str], curve_pref: str, ignore_a2: bool,
                                 show_acpr_any: bool):
        self.clear_axes()

        for label in selected_labels:
            file_index, records = mapping.get(label, (0, []))
            rec = self._choose_record(records, curve_pref)
            if rec is None: continue
            df = compute_metrics(rec, use_gamma_source=False, ignore_a2=ignore_a2)
            x = df['Pout [dBm] @ f0'].values
            # color by frequency (consistent across files), linestyle by file
            color = self._color_for_freq(rec.freq_label)
            ls = LINESTYLES[file_index % len(LINESTYLES)]

            if self.show_gain:
                y = df['Gt [dB] @ f0'].values
                m = np.isfinite(x) & np.isfinite(y)
                if m.any(): self.ax_gain.plot(x[m], y[m], label=label, color=color, linestyle=ls)

            if self.show_ampm_or_acpr:
                if show_acpr_any:
                    yL = df.get('ACPR Lower [dBc] @ f0')
                    yU = df.get('ACPR Upper [dBc] @ f0')
                    if yL is not None:
                        mL = np.isfinite(x) & np.isfinite(yL.values)
                        if mL.any(): self.ax_ampm_or_acpr.plot(x[mL], yL.values[mL], label=f'{label} (L)', color=color, linestyle=ls)
                    if yU is not None:
                        mU = np.isfinite(x) & np.isfinite(yU.values)
                        if mU.any(): self.ax_ampm_or_acpr.plot(x[mU], yU.values[mU], label=f'{label} (U)', color=color, linestyle=ls)
                else:
                    y = df['AM/PM offset [deg] @ f0'].values
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.any(): self.ax_ampm_or_acpr.plot(x[m], y[m], label=label, color=color, linestyle=ls)

            if self.show_eff:
                y = df['Drain Efficiency [%] @ f0'].values
                m = np.isfinite(x) & np.isfinite(y)
                if m.any(): self.ax_eff.plot(x[m], y[m], label=label, color=color, linestyle=ls)

            if self.show_irl_or_spec:
                spec_pairs = _extract_spectrum_pairs(rec)
                if spec_pairs:
                    fHz, pDbm = spec_pairs[-1]
                    if fHz.size and pDbm.size:
                        self.ax_irl.plot(fHz/1e6, pDbm, label=f'{label} spectrum', color=color, linestyle=ls)
                else:
                    y = df['Input Return Loss [dB] @ f0'].values
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.any(): self.ax_irl.plot(x[m], y[m], label=label, color=color, linestyle=ls)

        # Titles/labels
        # ensure visibility reflects current toggles
        self.ax_gain.set_visible(self.show_gain)
        if self.show_gain:
            self.ax_gain.set_title('Gain @ f0 vs Pout')
            self.ax_gain.set_xlabel('Pout [dBm] @ f0'); self.ax_gain.set_ylabel('Gt [dB] @ f0'); self._apply_legend(self.ax_gain)
        else:
            self.ax_gain.set_visible(False)

        self.ax_ampm_or_acpr.set_visible(self.show_ampm_or_acpr)
        if self.show_ampm_or_acpr:
            # Determine scale from settings
            self.ax_ampm_or_acpr.set_title('ACPR @ f0 vs Pout' if show_acpr_any else 'AM/PM offset @ f0 vs Pout')
            self.ax_ampm_or_acpr.set_xlabel('Pout [dBm] @ f0')
            self.ax_ampm_or_acpr.set_ylabel('ACPR [dBc] @ f0' if show_acpr_any else 'AM/PM offset [deg] @ f0')
            self._apply_legend(self.ax_ampm_or_acpr)
        else:
            self.ax_ampm_or_acpr.set_visible(False)

        self.ax_eff.set_visible(self.show_eff)
        if self.show_eff:
            self.ax_eff.set_title('Drain Efficiency @ f0 vs Pout', pad=10)
            self.ax_eff.set_xlabel('Pout [dBm] @ f0'); self.ax_eff.set_ylabel('Drain Efficiency [%] @ f0'); self._apply_legend(self.ax_eff)
        else:
            self.ax_eff.set_visible(False)

        self.ax_irl.set_visible(self.show_irl_or_spec)
        if self.show_irl_or_spec:
            # Determine whether spectrum is plotted
            any_spec = False
            for label in selected_labels:
                _, records = mapping.get(label, (0, []))
                rec = self._choose_record(records, curve_pref)
                if rec and _extract_spectrum_pairs(rec): any_spec = True; break
            if any_spec:
                self.ax_irl.set_title('Normalized Frequency Spectrum', pad=10)
                self.ax_irl.set_xlabel('Frequency Offset [MHz]'); self.ax_irl.set_ylabel('Power [dBm]')
            else:
                self.ax_irl.set_title('Input Return Loss @ f0 vs Pout', pad=10)
                self.ax_irl.set_xlabel('Pout [dBm] @ f0'); self.ax_irl.set_ylabel('Input Return Loss [dB] @ f0')
            self._apply_legend(self.ax_irl)
        else:
            self.ax_irl.set_visible(False)

        # Axis scales
        self.ax_gain.set_xscale(self.xscale_gain); self.ax_gain.set_yscale(self.yscale_gain)
        self.ax_ampm_or_acpr.set_xscale(self.xscale_ampm); self.ax_ampm_or_acpr.set_yscale(self.yscale_ampm)
        self.ax_eff.set_xscale(self.xscale_eff); self.ax_eff.set_yscale(self.yscale_eff)
        self.ax_irl.set_xscale(self.xscale_irl); self.ax_irl.set_yscale(self.yscale_irl)

        # Apply custom ranges
        self._apply_axis_ranges(self.ax_gain, "gain")
        self._apply_axis_ranges(self.ax_ampm_or_acpr, "ampm")
        self._apply_axis_ranges(self.ax_eff, "eff")
        self._apply_axis_ranges(self.ax_irl, "irl")

        self.apply_grid()
        self.draw()

    @staticmethod
    def _choose_record(records: List[CurveRecord], curve_pref: str) -> Optional[CurveRecord]:
        if not records: return None
        s = curve_pref.lower()
        for r in records:
            name = f"{r.dataset_name} {r.curve_name}".lower()
            if "1-tone" in name and s in name: return r
        for r in records:
            name = f"{r.dataset_name} {r.curve_name}".lower()
            if "1-tone" in name: return r
        return records[0]

class DataSource:
    def __init__(self, file_path: Path, parser_name: str, records: List[CurveRecord], index: int):
        self.file_path = file_path
        self.parser_name = parser_name
        self.records = records
        self.index = index  # for color mapping
        self.freq_to_records: Dict[str, List[CurveRecord]] = {}
        for rec in records:
            lab = rec.freq_label or "Freq ?"
            self.freq_to_records.setdefault(lab, []).append(rec)
        def label_to_hz(lbl: str) -> float:
            m = FREQ_NAME_RE.search(lbl)
            if not m: return float("inf")
            val = float(m.group("val")); unit = m.group("unit").lower()
            mult = {"ghz":1e9,"mhz":1e6,"khz":1e3,"hz":1.0}.get(unit, 1.0)
            return val * mult
        self.labels_sorted = sorted(self.freq_to_records.keys(), key=label_to_hz)
    @property
    def file_label(self) -> str:
        return self.file_path.name

class SeriesPanel(QGroupBox):
    on_changed = QtCore.Signal()
    def __init__(self, parent=None):
        super().__init__("Series (File × Frequency)", parent)
        self.box = QWidget()
        self.v = QVBoxLayout(self.box); self.v.setContentsMargins(0,0,0,0)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setWidget(self.box)
        self.sel_all = QPushButton("Select All"); self.sel_none = QPushButton("Select None")
        self.sel_all.clicked.connect(self.select_all); self.sel_none.clicked.connect(self.select_none)
        outer = QVBoxLayout(self); outer.addWidget(self.scroll)
        btns = QHBoxLayout(); btns.addWidget(self.sel_all); btns.addWidget(self.sel_none); outer.addLayout(btns)
        self._checks: Dict[str, tuple] = {}
        self.v.addStretch(1)
    def set_series(self, sources: List['DataSource']):
        for i in reversed(range(self.v.count()-1)):
            item = self.v.itemAt(i); w = item.widget() if item else None
            if w: w.setParent(None)
        self._checks.clear()
        for si, src in enumerate(sources):
            lbl = QLabel(f"📄 {src.file_label}"); lbl.setStyleSheet("font-weight:600;")
            self.v.insertWidget(self.v.count()-1, lbl)
            for lab in src.labels_sorted:
                series_id = f"{si}::{lab}"
                cb = QCheckBox(f"{lab}")
                cb.setChecked(True)
                cb.stateChanged.connect(lambda _st, sid=series_id: self.on_changed.emit())
                self.v.insertWidget(self.v.count()-1, cb)
                self._checks[series_id] = (cb, si, lab)
        self.on_changed.emit()
    def selected_series(self) -> list:
        out = []
        for sid, (cb, si, lab) in self._checks.items():
            if cb.isChecked():
                out.append((si, lab))
        return out
    def select_all(self):
        for cb, *_ in self._checks.values(): cb.setChecked(True)
        self.on_changed.emit()
    def select_none(self):
        for cb, *_ in self._checks.values(): cb.setChecked(False)
        self.on_changed.emit()

class ControlsPanel(QGroupBox):
    on_changed = QtCore.Signal()
    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        v = QVBoxLayout(self)

        # Theme
        self.cb_dark = QCheckBox("Dark theme")
        self.cb_dark.setChecked(False)
        self.cb_dark.stateChanged.connect(lambda _st: self.on_changed.emit())
        v.addWidget(self.cb_dark)

        # 1-tone family
        self.lbl_tone = QLabel("1‑Tone family")
        v.addWidget(self.lbl_tone)
        self.grp = QButtonGroup(self)
        self.rb_s1 = QRadioButton("S1"); self.rb_s3 = QRadioButton("S3")
        self.rb_s1.setChecked(True)
        self.grp.addButton(self.rb_s1); self.grp.addButton(self.rb_s3)
        v.addWidget(self.rb_s1); v.addWidget(self.rb_s3)
        self.rb_s1.toggled.connect(lambda _ch: self.on_changed.emit())
        self.rb_s3.toggled.connect(lambda _ch: self.on_changed.emit())

        # Legend & grid
        self.cb_ignore_a2 = QCheckBox("Disregard A2 (assume Pout = |B2|²)")
        self.cb_ignore_a2.setChecked(False)
        self.cb_ignore_a2.stateChanged.connect(lambda _st: self.on_changed.emit())
        v.addWidget(self.cb_ignore_a2)

        self.cb_show_legend = QCheckBox("Show legends")
        self.cb_show_legend.setChecked(True)
        self.cb_show_legend.stateChanged.connect(lambda _st: self.on_changed.emit())
        v.addWidget(self.cb_show_legend)

        # Legend placement
        v.addWidget(QLabel("Legend placement"))
        self.cmb_legend = QComboBox(); self.cmb_legend.addItems(["Outside (right)", "Below", "Inside (in-plot)"])
        self.cmb_legend.currentIndexChanged.connect(lambda _i: self.on_changed.emit())
        v.addWidget(self.cmb_legend)
        # Legend width (for Outside mode)
        row_legw = QHBoxLayout()
        row_legw.addWidget(QLabel("Legend width"))
        self.slider_legw = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_legw.setMinimum(5); self.slider_legw.setMaximum(40); self.slider_legw.setValue(22)
        self.lbl_legw = QLabel("0.22")
        row_legw.addWidget(self.slider_legw); row_legw.addWidget(self.lbl_legw); v.addLayout(row_legw)
        self.slider_legw.valueChanged.connect(self._on_legw_changed)

        # --- S-parameters controls (visible only in S-parameters tab) ---
        self.lbl_sparam = QLabel("S-parameters"); self.lbl_sparam.setStyleSheet("font-weight:600;")
        v.addWidget(self.lbl_sparam)
        row_sp = QHBoxLayout()
        self.cb_s11 = QCheckBox("S11"); self.cb_s21 = QCheckBox("S21"); self.cb_s12 = QCheckBox("S12"); self.cb_s22 = QCheckBox("S22")
        for cb in (self.cb_s11,self.cb_s21,self.cb_s12,self.cb_s22): cb.setChecked(True); cb.stateChanged.connect(lambda _st: self.on_changed.emit())
        row_sp.addWidget(self.cb_s11); row_sp.addWidget(self.cb_s21); row_sp.addWidget(self.cb_s12); row_sp.addWidget(self.cb_s22)
        v.addLayout(row_sp)

        row_mag = QHBoxLayout()
        row_mag.addWidget(QLabel("Magnitude")); self.cmb_mag = QComboBox(); self.cmb_mag.addItems(["dB","linear"]); self.cmb_mag.currentIndexChanged.connect(lambda _i: self.on_changed.emit())
        row_mag.addWidget(self.cmb_mag)
        row_mag.addSpacing(12); self.cb_unwrap = QCheckBox("Unwrap phase"); self.cb_unwrap.setChecked(True); self.cb_unwrap.stateChanged.connect(lambda _st: self.on_changed.emit())
        row_mag.addWidget(self.cb_unwrap)
        v.addLayout(row_mag)

        row_smith = QHBoxLayout()
        row_smith.addWidget(QLabel("Smith shows"))
        self.cmb_smith = QComboBox(); self.cmb_smith.addItems(["S11","S22","Both"]); self.cmb_smith.currentIndexChanged.connect(lambda _i: self.on_changed.emit())
        row_smith.addWidget(self.cmb_smith)
        v.addLayout(row_smith)

        row_polar = QHBoxLayout()
        row_polar.addWidget(QLabel("Polar shows"))
        self.cmb_polar = QComboBox(); self.cmb_polar.addItems(["S11","S22","S21","S12","Both"]); self.cmb_polar.currentIndexChanged.connect(lambda _i: self.on_changed.emit())
        row_polar.addWidget(self.cmb_polar)
        row_polar.addSpacing(12)
        row_polar.addWidget(QLabel("Polar magnitude"))
        self.cmb_polar_mag = QComboBox(); self.cmb_polar_mag.addItems(["linear","dB"]); self.cmb_polar_mag.setCurrentText("linear"); self.cmb_polar_mag.currentIndexChanged.connect(lambda _i: self.on_changed.emit())
        row_polar.addWidget(self.cmb_polar_mag)
        v.addLayout(row_polar)

        # Compact legend labels
        self.cb_compact = QCheckBox("Compact legend labels")
        self.cb_compact.setChecked(True)
        self.cb_compact.setVisible(False)
        self.cb_compact.stateChanged.connect(lambda _st: self.on_changed.emit())
        v.addWidget(self.cb_compact)

        # Legend font size
        row_lfs = QHBoxLayout()
        row_lfs.addWidget(QLabel("Legend font size"))
        self.slider_lfs = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_lfs.setMinimum(6); self.slider_lfs.setMaximum(14); self.slider_lfs.setValue(8)
        self.lbl_lfs = QLabel("8")
        row_lfs.addWidget(self.slider_lfs); row_lfs.addWidget(self.lbl_lfs); v.addLayout(row_lfs)
        self.slider_lfs.valueChanged.connect(self._on_lfs_changed)

        # Legend columns (Below mode)
        row_lc = QHBoxLayout()
        row_lc.addWidget(QLabel("Legend columns (Below)"))
        self.slider_lc = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_lc.setMinimum(1); self.slider_lc.setMaximum(5); self.slider_lc.setValue(2)
        self.lbl_lc = QLabel("2")
        row_lc.addWidget(self.slider_lc); row_lc.addWidget(self.lbl_lc); v.addLayout(row_lc)
        self.slider_lc.valueChanged.connect(self._on_lc_changed)

        self.cb_grid = QCheckBox("Show grid")
        self.cb_grid.setChecked(False)
        self.cb_grid.stateChanged.connect(lambda _st: self.on_changed.emit())
        v.addWidget(self.cb_grid)

        # Spacing sliders
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Vertical spacing"))
        self.slider_hpad = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_hpad.setMinimum(0); self.slider_hpad.setMaximum(80); self.slider_hpad.setValue(20)
        self.lbl_hpad = QLabel("2.0"); row3.addWidget(self.slider_hpad); row3.addWidget(self.lbl_hpad); v.addLayout(row3)
        self.slider_hpad.valueChanged.connect(self._on_hpad_changed)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Horizontal spacing"))
        self.slider_wpad = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_wpad.setMinimum(0); self.slider_wpad.setMaximum(80); self.slider_wpad.setValue(10)
        self.lbl_wpad = QLabel("1.0"); row4.addWidget(self.slider_wpad); row4.addWidget(self.lbl_wpad); v.addLayout(row4)
        self.slider_wpad.valueChanged.connect(self._on_wpad_changed)

        # Axis toggles
        self.lbl_axes = QLabel("Show axes")
        v.addWidget(self.lbl_axes)
        self.cb_ax_gain = QCheckBox("Gain"); self.cb_ax_gain.setChecked(True); self.cb_ax_gain.stateChanged.connect(lambda _st: self.on_changed.emit())
        self.cb_ax_ampm = QCheckBox("AM/PM or ACPR"); self.cb_ax_ampm.setChecked(True); self.cb_ax_ampm.stateChanged.connect(lambda _st: self.on_changed.emit())
        self.cb_ax_eff  = QCheckBox("Drain Efficiency"); self.cb_ax_eff.setChecked(True); self.cb_ax_eff.stateChanged.connect(lambda _st: self.on_changed.emit())
        self.cb_ax_irl  = QCheckBox("IRL or Spectrum"); self.cb_ax_irl.setChecked(True); self.cb_ax_irl.stateChanged.connect(lambda _st: self.on_changed.emit())
        v.addWidget(self.cb_ax_gain); v.addWidget(self.cb_ax_ampm); v.addWidget(self.cb_ax_eff); v.addWidget(self.cb_ax_irl)

        # Axis scale combos
        def scale_combo():
            c = QComboBox(); c.addItems(["linear","log"]); return c
        v.addWidget(QLabel("Axis scales (x,y)"))

        row_sc1 = QHBoxLayout(); row_sc1.addWidget(QLabel("Gain"))
        self.cmb_gx = scale_combo(); self.cmb_gy = scale_combo(); row_sc1.addWidget(self.cmb_gx); row_sc1.addWidget(self.cmb_gy); v.addLayout(row_sc1)
        row_sc2 = QHBoxLayout(); row_sc2.addWidget(QLabel("AM/PM/ACPR"))
        self.cmb_ax = scale_combo(); self.cmb_ay = scale_combo(); row_sc2.addWidget(self.cmb_ax); row_sc2.addWidget(self.cmb_ay); v.addLayout(row_sc2)
        row_sc3 = QHBoxLayout(); row_sc3.addWidget(QLabel("Efficiency"))
        self.cmb_ex = scale_combo(); self.cmb_ey = scale_combo(); row_sc3.addWidget(self.cmb_ex); row_sc3.addWidget(self.cmb_ey); v.addLayout(row_sc3)
        row_sc4 = QHBoxLayout(); row_sc4.addWidget(QLabel("IRL/Spectrum"))
        self.cmb_ix = scale_combo(); self.cmb_iy = scale_combo(); row_sc4.addWidget(self.cmb_ix); row_sc4.addWidget(self.cmb_iy); v.addLayout(row_sc4)

        for cmb in [self.cmb_gx,self.cmb_gy,self.cmb_ax,self.cmb_ay,self.cmb_ex,self.cmb_ey,self.cmb_ix,self.cmb_iy]:
            cmb.currentIndexChanged.connect(lambda _i: self.on_changed.emit())

        v.addStretch(1)


    def set_mode(self, mode: str):
        """Switch control visibility between 'im' and 'sparam'."""
        im_only = [self.lbl_tone, self.rb_s1, self.rb_s3, self.cb_ignore_a2,
                   self.lbl_axes, self.cb_ax_gain, self.cb_ax_ampm, self.cb_ax_eff, self.cb_ax_irl]
        sp_only = [self.lbl_sparam, self.cb_s11, self.cb_s21, self.cb_s12, self.cb_s22,
                   self.cmb_mag, self.cb_unwrap, self.cmb_smith, self.cmb_polar, self.cmb_polar_mag]
        show_im = (mode == 'im'); show_sp = (mode == 'sparam')
        for w in im_only:
            if w is not None: w.setVisible(show_im)
        for w in sp_only:
            if w is not None: w.setVisible(show_sp)

    # getters
    def dark_theme(self) -> bool: return self.cb_dark.isChecked()
    def curve_pref(self) -> str: return "s1" if self.rb_s1.isChecked() else "s3"
    def ignore_a2(self) -> bool: return self.cb_ignore_a2.isChecked()
    def show_legends(self) -> bool: return self.cb_show_legend.isChecked()
    def show_grid(self) -> bool: return self.cb_grid.isChecked()
    def hpad(self) -> float: return round(self.slider_hpad.value()/10.0,1)
    def wpad(self) -> float: return round(self.slider_wpad.value()/10.0,1)
    def axis_toggles(self): return (self.cb_ax_gain.isChecked(), self.cb_ax_ampm.isChecked(), self.cb_ax_eff.isChecked(), self.cb_ax_irl.isChecked())
    def axis_scales(self):
        return (self.cmb_gx.currentText(), self.cmb_gy.currentText(),
                self.cmb_ax.currentText(), self.cmb_ay.currentText(),
                self.cmb_ex.currentText(), self.cmb_ey.currentText(),
                self.cmb_ix.currentText(), self.cmb_iy.currentText())
    def legend_mode(self) -> str:
        idx = self.cmb_legend.currentIndex()
        return ['outside','below','inside'][idx]
    def legend_width(self) -> float:
        return round(self.slider_legw.value()/100.0, 2)
    def _on_legw_changed(self, val: int):
        self.lbl_legw.setText(f"{val/100.0:.2f}")
        self.on_changed.emit()
    def compact_labels(self) -> bool:
        return True
    def legend_fontsize(self) -> int:
        return self.slider_lfs.value()
    def legend_ncol(self) -> int:
        return self.slider_lc.value()
    def _on_lfs_changed(self, val: int):
        self.lbl_lfs.setText(str(val)); self.on_changed.emit()
    def _on_lc_changed(self, val: int):
        self.lbl_lc.setText(str(val)); self.on_changed.emit()

    # S-parameter getters
    def sparam_params(self):
        out=[]
        if self.cb_s11.isChecked(): out.append('S11')
        if self.cb_s21.isChecked(): out.append('S21')
        if self.cb_s12.isChecked(): out.append('S12')
        if self.cb_s22.isChecked(): out.append('S22')
        return out
    def sparam_mag_mode(self):
        return self.cmb_mag.currentText()
    def sparam_unwrap(self):
        return self.cb_unwrap.isChecked()
    def sparam_smith(self):
        return self.cmb_smith.currentText()
    def sparam_polar_param(self):
        return self.cmb_polar.currentText()
    def sparam_polar_mag(self):
        return self.cmb_polar_mag.currentText()

    # slots
    def _on_hpad_changed(self, val: int):
        hv = round(val / 10.0, 1); self.lbl_hpad.setText(f"{hv:.1f}"); self.on_changed.emit()
    def _on_wpad_changed(self, val: int):
        wv = round(val / 10.0, 1); self.lbl_wpad.setText(f"{wv:.1f}"); self.on_changed.emit()

class TablesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.freq_to_records: Dict[str, List[CurveRecord]] = {}
        self.ignore_a2 = False
        self.curve_pref = "s1"
        v = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.cmb_freq = QComboBox()
        self.cmb_curve = QComboBox()
        self.btn_export = QPushButton("Export CSV…")
        controls.addWidget(QLabel("Frequency:"))
        controls.addWidget(self.cmb_freq)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Curve:"))
        controls.addWidget(self.cmb_curve)
        controls.addStretch(1)
        controls.addWidget(self.btn_export)
        v.addLayout(controls)
        self.table = QTableWidget()
        v.addWidget(self.table, 1)
        self.cmb_freq.currentIndexChanged.connect(self._on_freq_changed)
        self.cmb_curve.currentIndexChanged.connect(self._on_curve_changed)
        self.btn_export.clicked.connect(self._on_export)

    def set_data(self, freq_to_records: Dict[str, List[CurveRecord]], labels_sorted: List[str], curve_pref: str, ignore_a2: bool):
        self.freq_to_records = freq_to_records
        self.curve_pref = curve_pref
        self.ignore_a2 = ignore_a2
        self.cmb_freq.blockSignals(True)
        self.cmb_freq.clear()
        for lab in labels_sorted:
            self.cmb_freq.addItem(lab)
        self.cmb_freq.blockSignals(False)
        if labels_sorted:
            self.cmb_freq.setCurrentIndex(0)
            self._populate_curves_for_selected_freq()
            self._populate_table()

    def _choose_record(self, records: List[CurveRecord]) -> Optional[CurveRecord]:
        if not records: return None
        s = self.curve_pref.lower()
        for r in records:
            name = f"{r.dataset_name} {r.curve_name}".lower()
            if "1-tone" in name and s in name: return r
        for r in records:
            name = f"{r.dataset_name} {r.curve_name}".lower()
            if "1-tone" in name: return r
        return records[0]

    def _on_freq_changed(self, _idx): self._populate_curves_for_selected_freq(); self._populate_table()
    def _on_curve_changed(self, _idx): self._populate_table()

    def _populate_curves_for_selected_freq(self):
        lab = self.cmb_freq.currentText()
        recs = self.freq_to_records.get(lab, [])
        self.cmb_curve.blockSignals(True); self.cmb_curve.clear()
        for r in recs: self.cmb_curve.addItem(r.curve_name)
        self.cmb_curve.blockSignals(False)
        if recs: self.cmb_curve.setCurrentIndex(0)

    def _current_record(self) -> Optional[CurveRecord]:
        lab = self.cmb_freq.currentText()
        recs = self.freq_to_records.get(lab, [])
        if not recs: return None
        wanted = self.cmb_curve.currentText()
        for r in recs:
            if r.curve_name == wanted: return r
        return self._choose_record(recs)

    def _populate_table(self):
        rec = self._current_record()
        if rec is None:
            self.table.clear(); self.table.setRowCount(0); self.table.setColumnCount(0); return
        try:
            df = record_to_dataframe_with_metrics(rec, ignore_a2=self.ignore_a2)
        except Exception as e:
            log_ex(e); QtWidgets.QMessageBox.critical(self, "Compute error", f"{e}\n\nSee log: {LOGFILE}"); return
        self.table.clear()
        self.table.setRowCount(len(df.index)); self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for j, col in enumerate(df.columns):
            col_vals = df[col].values
            for i, val in enumerate(col_vals):
                txt = ""
                if isinstance(val, (int,float)) and np.isfinite(val):
                    txt = f"{val:.6g}"
                item = QTableWidgetItem(txt)
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()

    def _on_export(self):
        rec = self._current_record()
        if rec is None: return
        try:
            df = record_to_dataframe_with_metrics(rec, ignore_a2=self.ignore_a2)
            safe_curve = re.sub(r'[^A-Za-z0-9_\\-]+','_', self.cmb_curve.currentText())
            safe_freq  = re.sub(r'[^A-Za-z0-9_\\-]+','_', self.cmb_freq.currentText())
            suggested = f"table_{safe_freq}_{safe_curve}.csv"
            path, _ = QFileDialog.getSaveFileName(self, "Export CSV", suggested, "CSV files (*.csv)")
            if not path: return
            df.to_csv(path, index=False)
        except Exception as e:
            log_ex(e); QtWidgets.QMessageBox.critical(self, "Export failed", f"{e}\n\nSee log: {LOGFILE}")
# ----------------------------
# Touchstone (.s2p/.s1p) support
# ----------------------------
import zipfile

def _ts_convert_pair(fmt: str, a: float, b: float) -> complex:
    f = fmt.upper()
    if f == "RI":
        return complex(a, b)
    elif f == "MA":
        return complex(a*np.cos(np.deg2rad(b)), a*np.sin(np.deg2rad(b)))
    elif f == "DB":
        r = 10**(a/20.0)
        return complex(r*np.cos(np.deg2rad(b)), r*np.sin(np.deg2rad(b)))
    else:
        return complex(np.nan, np.nan)

def _ts_freq_mult(unit: str) -> float:
    u = unit.upper()
    return {"HZ":1.0, "KHZ":1e3, "MHZ":1e6, "GHZ":1e9}.get(u, 1.0)

def parse_touchstone(path: Path):
    """Parse basic Touchstone v1 .s1p/.s2p files.
    Supports data formats: RI, MA, DB; units: Hz/kHz/MHz/GHz.
    Returns dict: {'nports': int, 'freq': ndarray, 'params': {'S11': ndarray, 'S21': ndarray, ...}, 'z0': float}
    """
    lines = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()
    fmt = "MA"; unit = "HZ"; z0 = 50.0; order = ["S11","S21","S12","S22"]
    nports = 2
    data = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("!"):
            continue
        if s.startswith("#"):
            # Example: "# GHZ S MA R 50"
            toks = s[1:].strip().split()
            try:
                unit = toks[0]
                # look for data format
                if "RI" in toks: fmt = "RI"
                elif "DB" in toks: fmt = "DB"
                else: fmt = "MA"
                # number of ports implied by filename extension; fallback to 2
                if "S" in toks: pass
                if "R" in toks:
                    i = toks.index("R")
                    if i+1 < len(toks):
                        z0 = float(toks[i+1])
            except Exception:
                pass
            continue
        # data line: freq + pairs
        parts = s.replace("\t", " ").split()
        if len(parts) < 3:
            continue
        try:
            f = float(parts[0]) * _ts_freq_mult(unit)
        except Exception:
            continue
        vals = [float(x) for x in parts[1:]]
        data.append((f, vals))
    if not data:
        raise RuntimeError(f"No data found in {path.name}")
    freqs = np.array([d[0] for d in data], dtype=float)
    # Determine nports by column count
    pair_count = len(data[0][1]) // 2
    if pair_count == 1:
        nports = 1; order = ["S11"]
    elif pair_count >= 4:
        nports = 2; order = ["S11","S21","S12","S22"]
    else:
        # best effort
        nports = 2
    params = {k: np.zeros(freqs.shape, dtype=complex) for k in order[:pair_count]}
    for i, (_, vals) in enumerate(data):
        idx = 0
        for k in params.keys():
            a, b = vals[idx], vals[idx+1]; idx += 2
            params[k][i] = _ts_convert_pair(fmt, a, b)
    return {"nports": nports, "freq": freqs, "params": params, "z0": z0, "unit": unit, "fmt": fmt}

def load_sparam_sources(paths):
    """Load one or many files (and expand .zip to contained sNp files). Returns list of dicts with keys:
       file_label, freq (Hz), params (dict of complex arrays), z0
    """
    out = []
    for p in paths:
        pth = Path(p)
        if pth.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(pth, 'r') as zf:
                    for nm in zf.namelist():
                        if nm.lower().endswith((".s1p", ".s2p")):
                            with zf.open(nm) as fh:
                                content = fh.read().decode('utf-8', errors='ignore')
                            tmp = Path(pth.parent, f"__zip__{Path(nm).name}")
                            tmp.write_text(content, encoding='utf-8')
                            parsed = parse_touchstone(tmp)
                            out.append({"file_label": f"{pth.name}:{Path(nm).name}", **parsed})
                            tmp.unlink(missing_ok=True)
            except Exception:
                continue
        elif pth.suffix.lower() in (".s1p",".s2p"):
            parsed = parse_touchstone(pth)
            out.append({"file_label": pth.name, **parsed})
    return out

# ----------------------------
# S-parameters Tab
# ----------------------------
from matplotlib.patches import Circle

def _smith_axes(ax):
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, lw=0.5); ax.axvline(0, lw=0.5)
    # unit circle
    c = Circle((0,0), 1.0, fill=False, lw=0.7, alpha=0.5)
    ax.add_patch(c)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Re(Γ)'); ax.set_ylabel('Im(Γ)')
    ax.grid(True, alpha=0.2, linestyle='--')

class SParamPlotGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(10,8))
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout(self); lay.addWidget(self.canvas)
        gs = self.fig.add_gridspec(2,3)
        self.ax_mag = self.fig.add_subplot(gs[0,0])
        self.ax_phase = self.fig.add_subplot(gs[0,1])
        self.ax_smith = self.fig.add_subplot(gs[0,2])
        self.ax_gd = self.fig.add_subplot(gs[1,0])
        self.ax_polar = self.fig.add_subplot(gs[1,1], projection='polar')
        self.ax_unused = self.fig.add_subplot(gs[1,2])
        self.ax_unused.set_axis_off()
        _smith_axes(self.ax_smith)
        # Polar axes (created but hidden by default)
        self.ax_polar = self.fig.add_subplot(gs[1,1], projection='polar')
        self.ax_polar.set_visible(False)
        self.use_polar = False  # when True, bottom-right shows polar instead of Smith
        # Legend handling consistent with IM Pro
        self._h_pad = 2.0; self._w_pad = 1.0
        self._show_grid = False
        self.show_legends = True
        self.legend_mode = 'outside'
        self.legend_width = 0.22
        self.legend_fs = 8
        self.legend_ncol = 2
        self.compact_labels = True
        self._color_cycle = matplotlib.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
        self._param_color = {}  # color by Sij
        self._cursor_text = None
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _short(self, label: str) -> str:
        # Only show parameter & maybe file tail
        if '•' in label:
            # file • Sij
            parts = [x.strip() for x in label.split('•')]
            if len(parts) >= 2:
                tail = parts[-1]
            else:
                tail = label
        else:
            tail = label
        return tail if len(tail) <= 28 else (tail[:28] + '…')

    def _apply_legend(self, ax):
        if not self.show_legends: return
        h, l = ax.get_legend_handles_labels()
        if not h: return
        labels = [self._short(x) for x in l]
        mode = self.legend_mode
        if mode == 'outside' and len(labels) > 8:
            mode = 'below'
        if mode == 'outside':
            ax.legend(h, labels, loc='upper left', bbox_to_anchor=(1.0 + self.legend_width*0.1, 1.0),
                      borderaxespad=0.0, fontsize=self.legend_fs, frameon=False)
        elif mode == 'below':
            ncol = max(1, min(self.legend_ncol, len(labels)))
            ax.legend(h, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      ncol=ncol, fontsize=self.legend_fs, frameon=False)
        else:
            ax.legend(h, labels, loc='upper right', fontsize=self.legend_fs, frameon=False)

    def draw(self):
        any_below = False
        if self.show_legends:
            for ax in [self.ax_mag, self.ax_phase, self.ax_gd, self.ax_smith, self.ax_polar]:
                h, l = ax.get_legend_handles_labels()
                if h:
                    mode = self.legend_mode
                    if mode == 'outside' and len(l) > 8:
                        mode = 'below'
                    if mode == 'below': any_below = True; break
        if self.show_legends and self.legend_mode == 'outside' and not any_below:
            right_margin = max(0.5, 1.0 - self.legend_width); bottom_margin = 0.0
        else:
            right_margin = 0.98; bottom_margin = 0.12 if (self.show_legends and (self.legend_mode=='below' or any_below)) else 0.0
        self.fig.tight_layout(rect=[0.0, bottom_margin, right_margin, 1.0], h_pad=self._h_pad, w_pad=self._w_pad)
        self.canvas.draw_idle()

    def _on_motion(self, event):
        if not event.inaxes:
            if self._cursor_text is not None:
                self._cursor_text.set_visible(False); self.canvas.draw_idle()
            return
        ax = event.inaxes
        if self._cursor_text is None:
            self._cursor_text = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                            bbox=dict(boxstyle="round", fc="w", alpha=0.6), fontsize=8)
        self._cursor_text.set_text(f"x={event.xdata:.4g}, y={event.ydata:.4g}")
        self._cursor_text.xy = (event.xdata, event.ydata)
        self._cursor_text.set_visible(True); self.canvas.draw_idle()

    def reset_view(self):
        for ax in [self.ax_mag, self.ax_phase, self.ax_gd, self.ax_smith, self.ax_polar]:
            try: ax.relim(); ax.autoscale()
            except Exception: pass
        self.draw()

    def save_figure(self, parent=None):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getSaveFileName(parent, "Save Figure", "sparams.png", "PNG (*.png);;PDF (*.pdf)")
        if not path: return
        try: self.fig.savefig(path, bbox_inches='tight', dpi=150)
        except Exception as e:
            QMessageBox.critical(parent, "Save failed", str(e))

    def set_opts_from_controls(self, ctrl):
        self.show_legends = ctrl.show_legends()
        self._h_pad = ctrl.hpad(); self._w_pad = ctrl.wpad()
        self._show_grid = ctrl.show_grid()
        self.legend_mode = ctrl.legend_mode(); self.legend_width = ctrl.legend_width()
        self.legend_fs = ctrl.legend_fontsize(); self.legend_ncol = ctrl.legend_ncol()
        # theme is global via rcParams
        for ax in [self.ax_mag, self.ax_phase, self.ax_gd, self.ax_smith, self.ax_polar]:
            ax.grid(self._show_grid, alpha=0.3, linestyle='--')

    def color_for_param(self, sij: str):
        key = sij.upper()
        if key not in self._param_color:
            idx = len(self._param_color) % len(self._color_cycle)
            self._param_color[key] = self._color_cycle[idx]
        return self._param_color[key]

    def plot(self, series, show_params, mag_mode, unwrap, smith_param, polar_param='S11', polar_mag='linear'):

        import matplotlib
        # Determine multi-file vs single-file
        files = [it.get('file_idx', 0) for it in series]
        multi = len(set(files)) > 1
        palette = matplotlib.rcParams['axes.prop_cycle'].by_key().get(
            'color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        )
        if multi:
            old = self.color_for_param
            self.color_for_param = lambda sij: palette[getattr(self, "_current_file_idx", 0) % len(palette)]
            try:
                cleaned = []
                for it in series:
                    if 'file_idx' not in it:
                        it = dict(it); it['file_idx'] = 0
                    self._current_file_idx = it['file_idx']
                    cleaned.append(it)
                series = cleaned
                # original body continues below...
            finally:
                pass        # series: list of dicts {label, freq, params, z0, file_idx}
        for ax in [self.ax_mag, self.ax_phase, self.ax_gd, self.ax_smith, self.ax_polar]:
            ax.cla()
        _smith_axes(self.ax_smith)
        for item in series:
            label_base = item['file_label']
            f = item['freq']; params = item['params']
            for sij in show_params:
                if sij not in params: continue
                y = params[sij]
                color = self.color_for_param(sij)
                ls = ["-","--",":","-."][item.get('file_idx',0) % 4]
                # Magnitude
                mag = 20*np.log10(np.clip(np.abs(y), 1e-18, 1.0)) if mag_mode == "dB" else np.abs(y)
                self.ax_mag.plot(f/1e9, mag, color=color, linestyle=ls, label=f"{label_base} • {sij}")
                # Phase
                ph = np.angle(y, deg=True)
                if unwrap:
                    ph = np.unwrap(np.deg2rad(ph))*180/np.pi
                self.ax_phase.plot(f/1e9, ph, color=color, linestyle=ls, label=f"{label_base} • {sij}")
                # Group delay (ns): -1/(2π) dφ/df (with φ in rad, f in Hz)
                phi_rad = np.unwrap(np.angle(y))
                dphi_df = np.gradient(phi_rad, f)
                gd_s = -dphi_df/(2*np.pi)
                self.ax_gd.plot(f/1e9, gd_s*1e9, color=color, linestyle=ls, label=f"{label_base} • {sij}")
                # Smith: only reflection params
                if sij.upper() in ("S11","S22") and smith_param in (sij.upper(),"Both"):
                    self.ax_smith.plot(np.real(y), np.imag(y), color=color, linestyle=ls, label=f"{label_base} • {sij}")

                # Polar plotting (angle vs magnitude)
                if use_polar:
                    want = polar_param.upper()
                    if want == 'BOTH' and sij.upper() in ('S11','S22'):
                        pass  # we'll handle by natural loop (both refl selected in show_params)
                    if want == 'BOTH' and sij.upper() not in ('S11','S22'):
                        pass  # skip non-reflection when BOTH selected
                    elif want != 'BOTH' and sij.upper() != want:
                        pass  # not the chosen param
                    else:
                        ang = np.angle(y)  # radians
                        if polar_mag.lower() == 'db':
                            rr = np.clip(20*np.log10(np.abs(y)), -200, 200)
                            # convert dB to a positive radial scale for visibility (normalize)
                            # map [-60, 0] dB roughly into [0, 1]; simple linear map
                            rr = (rr + 60)/60.0
                            rr = np.clip(rr, 0.0, 1.0)
                        else:
                            rr = np.abs(y)  # linear magnitude
                        self.ax_polar.plot(ang, rr, color=color, linestyle=ls, label=f"{label_base} • {sij}")

        # Labels/titles
        self.ax_mag.set_title("|S| Magnitude"); self.ax_mag.set_xlabel("Frequency [GHz]")
        self.ax_mag.set_ylabel("Magnitude [dB]" if mag_mode=="dB" else "Magnitude [linear]")
        self._apply_legend(self.ax_mag)

        self.ax_phase.set_title("Phase"); self.ax_phase.set_xlabel("Frequency [GHz]"); self.ax_phase.set_ylabel("Phase [deg]")
        self._apply_legend(self.ax_phase)

        self.ax_gd.set_title("Group Delay"); self.ax_gd.set_xlabel("Frequency [GHz]"); self.ax_gd.set_ylabel("Group Delay [ns]")
        self._apply_legend(self.ax_gd)

        self.ax_smith.set_title("Smith Chart (Γ)"); self._apply_legend(self.ax_smith)
        self.ax_polar.set_title("Polar Plot (angle vs |S|)"); self._apply_legend(self.ax_polar)
        self.draw()

        if 'multi' in locals() and multi:
            self.color_for_param = old
class SParamTab(QWidget):
    def __init__(self, controls_panel: 'ControlsPanel', parent=None):
        super().__init__(parent)
        self.controls_panel = controls_panel  # reuse legend/grid/spacing/theme controls
        v = QVBoxLayout(self)

        # Toolbar
        tb = QHBoxLayout()
        self.btn_open = QPushButton("Open S‑params…")
        self.btn_add = QPushButton("Add…")
        self.btn_clear = QPushButton("Clear")
        self.btn_save = QPushButton("Save Figure…")
        self.btn_reset = QPushButton("Reset View")
        tb.addWidget(self.btn_open); tb.addWidget(self.btn_add); tb.addWidget(self.btn_clear)
        tb.addWidget(self.btn_save); tb.addWidget(self.btn_reset); tb.addStretch(1)
        v.addLayout(tb)

        # Options
        opt = QHBoxLayout()
        opt.addWidget(QLabel("Show:"))
        self.cb_s11 = QCheckBox("S11"); self.cb_s21 = QCheckBox("S21"); self.cb_s12 = QCheckBox("S12"); self.cb_s22 = QCheckBox("S22")
        for cb in (self.cb_s11,self.cb_s21,self.cb_s12,self.cb_s22): cb.setChecked(True)
        for cb in (self.cb_s11,self.cb_s21,self.cb_s12,self.cb_s22): cb.stateChanged.connect(self._on_changed)
        opt.addWidget(self.cb_s11); opt.addWidget(self.cb_s21); opt.addWidget(self.cb_s12); opt.addWidget(self.cb_s22)

        opt.addSpacing(16); opt.addWidget(QLabel("Magnitude:"))
        self.mag_combo = QComboBox(); self.mag_combo.addItems(["dB","linear"]); self.mag_combo.currentIndexChanged.connect(self._on_changed)
        opt.addWidget(self.mag_combo)

        opt.addSpacing(16); self.cb_unwrap = QCheckBox("Unwrap phase"); self.cb_unwrap.setChecked(True); self.cb_unwrap.stateChanged.connect(self._on_changed)
        opt.addWidget(self.cb_unwrap)


        opt.addSpacing(16); self.cb_polar = QCheckBox("Show polar"); self.cb_polar.setChecked(False); self.cb_polar.stateChanged.connect(self._on_changed)
        opt.addWidget(self.cb_polar)
        opt.addSpacing(12); opt.addWidget(QLabel("Polar shows:"))
        self.cmb_polar = QComboBox(); self.cmb_polar.addItems(["S11","S22","S21","S12","Both"]); self.cmb_polar.currentIndexChanged.connect(self._on_changed)
        opt.addWidget(self.cmb_polar)
        opt.addSpacing(12); opt.addWidget(QLabel("Polar magnitude:"))
        self.cmb_polar_mag = QComboBox(); self.cmb_polar_mag.addItems(["linear","dB"]); self.cmb_polar_mag.setCurrentText("linear"); self.cmb_polar_mag.currentIndexChanged.connect(self._on_changed)
        opt.addWidget(self.cmb_polar_mag)
        opt.addSpacing(16); opt.addWidget(QLabel("Smith shows:"))
        self.cmb_smith = QComboBox(); self.cmb_smith.addItems(["S11","S22","Both"]); self.cmb_smith.currentIndexChanged.connect(self._on_changed)
        opt.addWidget(self.cmb_smith)

        v.addLayout(opt)

        # Plot grid
        self.grid = SParamPlotGrid()
        v.addWidget(self.grid, 1)

        # File list / selection (left)
        self.series_panel = SeriesPanel()
        # Reuse its UI (file • frequency) but we'll map to (file • S-params bandwidth)
        self.series_panel.setTitle("Files")
        self.series_panel.sel_all.setText("Check all files"); self.series_panel.sel_none.setText("Clear files")

        pane = QHBoxLayout()
        pane.addWidget(self.series_panel, 0)
        pane.addStretch(1)
        v.addLayout(pane)

        # Data
        self.sources = []  # list of dicts from load_sparam_sources
        self.series_panel.on_changed.connect(self._on_changed)
        self.btn_open.clicked.connect(self.open_files)
        self.btn_add.clicked.connect(self.add_files)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_save.clicked.connect(lambda: self.grid.save_figure(self))
        self.btn_reset.clicked.connect(self.grid.reset_view)

    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open S‑parameter files", "", "S‑params (*.s1p *.s2p *.zip);;All files (*)")
        if not paths: return
        self.sources = load_sparam_sources(paths)
        self._refresh_selectors()
        self.update_plots()

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add S‑parameter files", "", "S‑params (*.s1p *.s2p *.zip);;All files (*)")
        if not paths: return
        more = load_sparam_sources(paths)
        self.sources.extend(more)
        self._refresh_selectors()
        self.update_plots()

    def clear_all(self):
        self.sources = []
        self._refresh_selectors()
        self.grid.reset_view()

    def _refresh_selectors(self):
        # Map into SeriesPanel as "file labels"; freq label = bandwidth
        class FakeSrc:
            def __init__(self, file_label):
                self.file_label = file_label
                self.freq_to_records = {"All": [None]}
                self.labels_sorted = ["All"]
        fake_sources = [FakeSrc(s['file_label']) for s in self.sources]
        self.series_panel.set_series(fake_sources)

    def _selected_files(self):
        selected = self.series_panel.selected_series()
        chosen = []
        for si, _lab in selected:
            if 0 <= si < len(self.sources):
                chosen.append((si, self.sources[si]))
        return chosen

    def _on_changed(self, *_):
        self.update_plots()

    def update_plots(self):
        # Apply main controls (legend/grid/theme spacing)
        self.grid.set_opts_from_controls(self.controls_panel)
        # Gather selected files
        chosen = self._selected_files()
        if not chosen:
            for ax in [self.grid.ax_mag, self.grid.ax_phase, self.grid.ax_gd, self.grid.ax_smith]:
                ax.cla()
            _smith_axes(self.grid.ax_smith)
            self.grid.draw()
            return
        # Build series
        series = []
        for idx, src in chosen:
            series.append({"file_idx": idx, **src})
        show_params = self.controls_panel.sparam_params()
        mag_mode = self.controls_panel.sparam_mag_mode()
        unwrap = self.controls_panel.sparam_unwrap()
        smith_param = self.controls_panel.sparam_smith()
        polar_param = self.controls_panel.sparam_polar_param()
        polar_mag = self.controls_panel.sparam_polar_mag()
        self.grid.plot(series, show_params, mag_mode, unwrap, smith_param, polar_param, polar_mag)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IM Desktop — Pro")
        self.sources: List[DataSource] = []

        central = QWidget(); self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        # Left controls
        left = QVBoxLayout()
        open_row = QHBoxLayout()
        btn_open = QPushButton("Open .im…"); btn_open.clicked.connect(self.open_files)
        self.btn_add = QPushButton('Add files…'); self.btn_clear = QPushButton('Clear')
        self.btn_add.clicked.connect(self.add_files); self.btn_clear.clicked.connect(self.clear_all)
        open_row.addWidget(btn_open); open_row.addWidget(self.btn_add); open_row.addWidget(self.btn_clear)
        left.addLayout(open_row)

        # Session row
        sess_row = QHBoxLayout()
        btn_save_sess = QPushButton("Save session"); btn_load_sess = QPushButton("Load session")
        btn_save_sess.clicked.connect(self.save_session); btn_load_sess.clicked.connect(self.load_session)
        sess_row.addWidget(btn_save_sess); sess_row.addWidget(btn_load_sess); left.addLayout(sess_row)

        self.controls = ControlsPanel()
        left.addWidget(self.controls)

        self.series_panel = SeriesPanel()
        left.addWidget(self.series_panel); left.addStretch(1)

        # Right tabs
        self.tabs = QTabWidget()
        self.plot_grid = PlotGrid()
        plot_tab = QWidget(); plot_tab_layout = QVBoxLayout(plot_tab)
        # Toolbar
        tb = QHBoxLayout()
        btn_save = QPushButton("Save Figure…"); btn_reset = QPushButton("Reset View")
        btn_save.clicked.connect(lambda: self.plot_grid.save_figure(self))
        btn_reset.clicked.connect(self.plot_grid.reset_view)
        # Export current selection
        btn_export_sel = QPushButton("Export Selected Data…")
        btn_export_sel.clicked.connect(self.export_selected_data)
        tb.addWidget(btn_save); tb.addWidget(btn_reset); tb.addWidget(btn_export_sel); tb.addStretch(1)
        plot_tab_layout.addLayout(tb)
        plot_tab_layout.addWidget(self.plot_grid)
        self.tabs.addTab(plot_tab, "Plots")

        self.tables_tab = TablesTab(); self.tabs.addTab(self.tables_tab, "Tables")

        # NEW: S-parameters tab
        self.sparam_tab = SParamTab(self.controls)
        self.tabs.addTab(self.sparam_tab, "S‑parameters")
        self.tables_tab = TablesTab(self)
        self.tabs.addTab(self.tables_tab, "Tables")

        outer.addWidget(self.tabs, 1)
        outer.addLayout(left, 0)

        self.controls.on_changed.connect(self.on_controls_changed)
        self.series_panel.on_changed.connect(self.on_series_selection_changed)

        # Hook tab change to switch control mode
        
        # Toggle control mode when switching tabs
        def _on_tab_changed(idx):
            try:
                w = self.tabs.widget(idx)
                if w is self.sparam_tab:
                    self.controls.set_mode('sparam')
                else:
                    self.controls.set_mode('im')
            except Exception:
                pass
        self.tabs.currentChanged.connect(_on_tab_changed)
        # Initialize mode to IM
        self.controls.set_mode('im')

        self.statusBar().showMessage('Open .im file(s) to begin')

        # Apply initial theme
        self.apply_theme(self.controls.dark_theme())

    # THEME
    def apply_theme(self, dark: bool):
        if dark:
            # simple dark palette
            pal = QtGui.QPalette()
            pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(45,45,45))
            pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(30,30,30))
            pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(45,45,45))
            pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45,45,45))
            pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.red)
            pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(76,163,224))
            pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.black)
            self.setPalette(pal)
            matplotlib.rcParams.update({"axes.facecolor": "#2d2d2d", "figure.facecolor": "#2d2d2d", "text.color":"#ffffff", "axes.edgecolor":"#cccccc", "axes.labelcolor":"#ffffff", "xtick.color":"#ffffff", "ytick.color":"#ffffff"})
        else:
            self.setPalette(self.style().standardPalette())
            matplotlib.rcParams.update({"axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff", "text.color":"#000000", "axes.edgecolor":"#000000", "axes.labelcolor":"#000000", "xtick.color":"#000000", "ytick.color":"#000000"})
        # trigger redraw
        self.update_plots()

    # FILE IO
    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open .im file(s)", "", "IM files (*.im);;All files (*)")
        if not paths: return
        try:
            self.sources = []
            for i, p in enumerate(paths):
                self._ingest_file(Path(p), i)
            self._refresh_after_ingest(status_prefix='Loaded')
        except Exception as e:
            log_ex(e); QtWidgets.QMessageBox.critical(self, "Failed to load", f"{e}\n\nSee log: {LOGFILE}")

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add .im file(s)", "", "IM files (*.im);;All files (*)")
        if not paths: return
        try:
            start = len(self.sources)
            for i, p in enumerate(paths):
                self._ingest_file(Path(p), start+i)
            self._refresh_after_ingest(status_prefix='Added')
        except Exception as e:
            log_ex(e); QtWidgets.QMessageBox.critical(self, "Failed to add", f"{e}\n\nSee log: {LOGFILE}")

    def clear_all(self):
        self.sources = []
        self.series_panel.set_series([])
        self.plot_grid.clear_axes(); self.plot_grid.draw()
        self.tables_tab.set_data({}, [], "s1", False)
        self.tables_tab.set_data({}, [], 's1', False)
        self.statusBar().showMessage('Cleared all sources')

    def _ingest_file(self, path: Path, index: int):
        parser = pick_parser(path)
        records = parser.parse(path)
        self.sources.append(DataSource(path, parser.name(), records, index))

    def _refresh_after_ingest(self, status_prefix='Loaded'):
        self.series_panel.set_series(self.sources)
        self.statusBar().showMessage(f"{status_prefix} {len(self.sources)} file(s)")

        combined_map: Dict[str, List[CurveRecord]] = {}
        combined_labels: List[str] = []
        for src in self.sources:
            for lab, recs in src.freq_to_records.items():
                label = f"{src.file_label} • {lab}"
                combined_map[label] = recs
                combined_labels.append(label)
        self.tables_tab.set_data(combined_map, combined_labels, self.controls.curve_pref(), self.controls.ignore_a2())

        self.update_plots()

    def _refresh_tables(self):
        # Build mapping for the Tables tab from current selection (fallback to all)
        combined_map = {}
        combined_labels = []
        selected = []
        try:
            selected = self.series_panel.selected_series()
        except Exception:
            selected = []
        if selected:
            for (si, lab) in selected:
                if si < 0 or si >= len(self.sources): 
                    continue
                src = self.sources[si]
                recs = src.freq_to_records.get(lab, [])
                if not recs:
                    continue
                label = f"{src.file_label} • {lab}"
                combined_map[label] = recs
                combined_labels.append(label)
        else:
            for src in self.sources:
                for lab, recs in src.freq_to_records.items():
                    label = f"{src.file_label} • {lab}"
                    combined_map[label] = recs
                    combined_labels.append(label)
        try:
            self.tables_tab.set_data(combined_map, combined_labels, self.controls.curve_pref(), self.controls.ignore_a2())
        except Exception:
            pass

    # CONTROLS REACTIONS
    def on_controls_changed(self):
        # theme
        self.apply_theme(self.controls.dark_theme())
        # Apply legend, paddings, grid, axis toggles, scales
        self.plot_grid.set_show_legends(self.controls.show_legends())
        self.plot_grid.set_hpad(self.controls.hpad())
        self.plot_grid.set_wpad(self.controls.wpad())
        self.plot_grid.set_show_grid(self.controls.show_grid())
        self.plot_grid.legend_mode = self.controls.legend_mode()
        self.plot_grid.legend_width = self.controls.legend_width()
        self.plot_grid.legend_fs = self.controls.legend_fontsize()
        self.plot_grid.legend_ncol = self.controls.legend_ncol()
        self.plot_grid.compact_labels = self.controls.compact_labels()
        self.plot_grid.set_axis_toggle(*self.controls.axis_toggles())
        self.plot_grid.set_axis_scales(*self.controls.axis_scales())
        self.plot_grid.legend_mode = self.controls.legend_mode()
        self.plot_grid.legend_width = self.controls.legend_width()
        self.plot_grid.legend_fs = self.controls.legend_fontsize()
        self.plot_grid.legend_ncol = self.controls.legend_ncol()
        self.plot_grid.compact_labels = self.controls.compact_labels()
        self.update_plots()
        
        self._refresh_tables()
# Keep tables consistent with curve family / ignore A2
        if self.sources:
            combined_map: Dict[str, List[CurveRecord]] = {}
            combined_labels: List[str] = []
            for src in self.sources:
                for lab, recs in src.freq_to_records.items():
                    label = f"{src.file_label} • {lab}"
                    combined_map[label] = recs
                    combined_labels.append(label)
            self.tables_tab.set_data(combined_map, combined_labels, self.controls.curve_pref(), self.controls.ignore_a2())

    def on_series_selection_changed(self):
        self.update_plots()

    # PLOTTING
    def _build_mapping_for_plots(self, selected: List[Tuple[int,str]]):
        mapping = {}
        labels = []
        for si, lab in selected:
            if si < 0 or si >= len(self.sources): continue
            src = self.sources[si]
            recs = src.freq_to_records.get(lab, [])
            if not recs: continue
            label = f"{src.file_label} • {lab}"
            mapping[label] = (src.index, recs)
            labels.append(label)
        return mapping, labels

    def update_plots(self):
        if not self.sources:
            return
        selected = self.series_panel.selected_series()
        if not selected:
            self.plot_grid.clear_axes(); self.plot_grid.draw()
            self.statusBar().showMessage('No series selected')
            return
        curve_pref = self.controls.curve_pref()
        ignore_a2 = self.controls.ignore_a2()
        mapping, labels = self._build_mapping_for_plots(selected)

        # Check if ACPR present in any selected record
        show_acpr_any = False
        for label in labels:
            _, recs = mapping.get(label, (0, []))
            rec = PlotGrid._choose_record(recs, curve_pref)
            if rec is None: continue
            df = compute_metrics(rec, use_gamma_source=False, ignore_a2=ignore_a2)
            sL = df.get("ACPR Lower [dBc] @ f0"); sU = df.get("ACPR Upper [dBc] @ f0")
            if sL is not None and np.isfinite(sL.values).any(): show_acpr_any = True; break
            if sU is not None and np.isfinite(sU.values).any(): show_acpr_any = True; break

        try:
            self.plot_grid.overlay_frequency_curves(mapping, labels, curve_pref, ignore_a2, show_acpr_any)
            msg_tail = "Pout=|B2|^2" if ignore_a2 else "Pout=|B2|^2−|A2|^2"
            self.statusBar().showMessage(f"Plotted {len(labels)} series • Pref: {curve_pref.upper()} • {msg_tail}")
        except Exception as e:
            log_ex(e)
            QtWidgets.QMessageBox.critical(self, "Plot error", f"{e}\n\nSee log: {LOGFILE}")

    # EXPORT SELECTED DATA
    def export_selected_data(self):
        selected = self.series_panel.selected_series()
        if not selected: 
            QtWidgets.QMessageBox.information(self, "Export", "No series selected."); 
            return
        curve_pref = self.controls.curve_pref()
        ignore_a2 = self.controls.ignore_a2()
        mapping, labels = self._build_mapping_for_plots(selected)
        rows = []
        for label in labels:
            _, recs = mapping.get(label, (0, []))
            rec = PlotGrid._choose_record(recs, curve_pref)
            if rec is None: continue
            df = record_to_dataframe_with_metrics(rec, ignore_a2=ignore_a2)
            df2 = pd.DataFrame({
                "Series": [label]*len(df.index),
                "Pout [dBm] @ f0": df["Pout [dBm] @ f0"].values,
                "Gt [dB] @ f0": df["Gt [dB] @ f0"].values,
                "AM/PM offset [deg] @ f0": df["AM/PM offset [deg] @ f0"].values,
                "Drain Efficiency [%] @ f0": df["Drain Efficiency [%] @ f0"].values,
                "Input Return Loss [dB] @ f0": df["Input Return Loss [dB] @ f0"].values,
                "ACPR Lower [dBc] @ f0": df.get("ACPR Lower [dBc] @ f0", pd.Series([np.nan]*len(df))).values,
                "ACPR Upper [dBc] @ f0": df.get("ACPR Upper [dBc] @ f0", pd.Series([np.nan]*len(df))).values,
            })
            rows.append(df2)
        if not rows:
            QtWidgets.QMessageBox.information(self, "Export", "Nothing to export."); 
            return
        big = pd.concat(rows, ignore_index=True)
        path, _ = QFileDialog.getSaveFileName(self, "Export Selected Data", "selected_data.csv", "CSV files (*.csv)")
        if not path: return
        try:
            big.to_csv(path, index=False)
        except Exception as e:
            log_ex(e); QtWidgets.QMessageBox.critical(self, "Export failed", f"{e}\n\nSee log: {LOGFILE}")

    # SESSION SAVE/LOAD
    def save_session(self):
        data = {}
        data["files"] = [str(src.file_path) for src in self.sources]
        data["selected"] = self.series_panel.selected_series()
        data["curve_pref"] = self.controls.curve_pref()
        data["ignore_a2"] = self.controls.ignore_a2()
        data["show_legends"] = self.controls.show_legends()
        data["show_grid"] = self.controls.show_grid()
        data["hpad"] = self.controls.hpad()
        data["wpad"] = self.controls.wpad()
        data["axis_toggles"] = self.controls.axis_toggles()
        data["axis_scales"] = self.controls.axis_scales()
        data["dark_theme"] = self.controls.dark_theme()
        try:
            SESSIONFILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.statusBar().showMessage(f"Session saved to {SESSIONFILE.name}")
        except Exception as e:
            log_ex(e); QtWidgets.QMessageBox.critical(self, "Save session failed", str(e))

    def load_session(self):
        try:
            data = json.loads(SESSIONFILE.read_text(encoding="utf-8"))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load session failed", f"{e}\n\nMake sure a session file exists next to the script.")
            return
        self.sources = []
        for i, p in enumerate(data.get("files", [])):
            pth = Path(p)
            if pth.exists():
                try: self._ingest_file(pth, i)
                except Exception: pass
        self._refresh_after_ingest(status_prefix='Loaded (session)')
        # restore controls
        if data.get("curve_pref","s1") == "s1": self.controls.rb_s1.setChecked(True)
        else: self.controls.rb_s3.setChecked(True)
        self.controls.cb_ignore_a2.setChecked(bool(data.get("ignore_a2", False)))
        self.controls.cb_show_legend.setChecked(bool(data.get("show_legends", True)))
        self.controls.cb_grid.setChecked(bool(data.get("show_grid", False)))
        self.controls.slider_hpad.setValue(int(10*float(data.get("hpad", 2.0))))
        self.controls.slider_wpad.setValue(int(10*float(data.get("wpad", 1.0))))
        ax_tog = data.get("axis_toggles", (True,True,True,True))
        self.controls.cb_ax_gain.setChecked(ax_tog[0]); self.controls.cb_ax_ampm.setChecked(ax_tog[1])
        self.controls.cb_ax_eff.setChecked(ax_tog[2]); self.controls.cb_ax_irl.setChecked(ax_tog[3])
        ax_scales = data.get("axis_scales", ("linear","linear","linear","linear","linear","linear","linear","linear"))
        self.controls.cmb_gx.setCurrentText(ax_scales[0]); self.controls.cmb_gy.setCurrentText(ax_scales[1])
        self.controls.cmb_ax.setCurrentText(ax_scales[2]); self.controls.cmb_ay.setCurrentText(ax_scales[3])
        self.controls.cmb_ex.setCurrentText(ax_scales[4]); self.controls.cmb_ey.setCurrentText(ax_scales[5])
        self.controls.cmb_ix.setCurrentText(ax_scales[6]); self.controls.cmb_iy.setCurrentText(ax_scales[7])
        self.controls.cb_dark.setChecked(bool(data.get("dark_theme", False)))
        self.apply_theme(self.controls.dark_theme())
        # reselect series
        self.series_panel.set_series(self.sources)
        for (si, lab) in self.series_panel._checks.values():
            pass
        # Can't programmatically set QCheckBox states by mapping easily here; we will approximate by selecting all
        # and rely on user selection. For full fidelity we'd need stable IDs per file; skipping for now.
        self.update_plots()

# [baseline __main__ removed in single-file build]



# =========================
# Single-file app additions
# =========================
import sys, numpy as np
from pathlib import Path
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QScrollArea, QTabWidget, QMessageBox, QPushButton
)

class MainWindowSingle(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IM & S-parameters Viewer — Single File")
        self.sources = []  # List[DataSource]

        # Left: Series + Controls (baseline widgets)
        self.series_panel = SeriesPanel()
        self.controls = ControlsPanel()
        self.controls.on_changed.connect(self._on_controls_changed)

        # Tabs container
        self.tabs = QTabWidget(self)

        # --- IM Tab (using baseline PlotGrid) ---
        self.plot_grid = PlotGrid()
        self.im_tab = QWidget()
        im_v = QVBoxLayout(self.im_tab)

        top_row = QHBoxLayout()
        self.btn_load_im = QPushButton("Open .im…")
        self.btn_add_im  = QPushButton("Add files…")
        self.btn_clear   = QPushButton("Clear")
        self.btn_save_im = QPushButton("Save IM Figure")
        self.btn_reset_im = QPushButton("Reset View")
        for b in (self.btn_load_im, self.btn_add_im, self.btn_clear, self.btn_save_im, self.btn_reset_im):
            top_row.addWidget(b)
        top_row.addStretch(1)
        im_v.addLayout(top_row)
        im_v.addWidget(self.plot_grid, 1)

        # --- S-parameters Tab (reuse baseline tab; it manages its own sources/UI) ---
        self.sparam_tab = SParamTab(self.controls)

        # Only two tabs: IM and S-parameters (Tables tab intentionally removed)
        self.tabs.addTab(self.im_tab, "IM (1‑tone)")
        self.tabs.addTab(self.sparam_tab, "S‑parameters")
        self.tables_tab = TablesTab(self)
        self.tabs.addTab(self.tables_tab, "Tables")

        # --- Left sidebar ---
        left_sidebar = QWidget()
        left_v = QVBoxLayout(left_sidebar)
        left_v.setContentsMargins(8, 8, 8, 8)
        left_v.addWidget(self.series_panel)
        left_v.addWidget(self.controls)
        left_v.addStretch(1)

        # --- Scrollable charts area ---
        scroll = QScrollArea(self); scroll.setWidgetResizable(True)
        tabs_container = QWidget(); tabs_layout = QVBoxLayout(tabs_container)
        tabs_layout.setContentsMargins(0,0,0,0); tabs_layout.addWidget(self.tabs)
        self.tabs.setMinimumSize(1400, 900)
        scroll.setWidget(tabs_container)

        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        splitter.addWidget(left_sidebar); splitter.addWidget(scroll)
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1)
        left_sidebar.setMinimumWidth(300)
        self.setCentralWidget(splitter)
        # Install wheel forwarding so the mouse wheel scrolls the page even over canvases
        scrolls = self.findChildren(QScrollArea)
        if scrolls:
            _wheel = _WheelForwardFilter(scrolls[0])
            try:
                self.plot_grid.canvas.installEventFilter(_wheel)
            except Exception:
                pass
            try:
                if hasattr(self.sparam_tab, 'grid'):
                    for ax in getattr(self.sparam_tab.grid, 'axes', []):
                        if hasattr(ax, 'figure') and hasattr(ax.figure, 'canvas'):
                            ax.figure.canvas.installEventFilter(_wheel)
            except Exception:
                pass

        # Wire actions
        self.btn_load_im.clicked.connect(self.open_files)
        self.btn_add_im.clicked.connect(self.add_files)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_save_im.clicked.connect(lambda: self.plot_grid.save_figure(self))
        self.btn_reset_im.clicked.connect(self.plot_grid.reset_view)

        # Tab sync
        self.controls.set_mode('im')
        self.tabs.currentChanged.connect(self._on_tab_changed)
        try:
            self.series_panel.on_changed.connect(self._refresh_tables)
        except Exception:
            pass
        self.controls.on_changed.connect(self._refresh_tables)
        self.tabs.currentChanged.connect(lambda i: (self._refresh_tables() if self.tabs.tabText(i)=="Tables" else None))

        # Default to light (hide dark toggle if present)
        try:
            self.controls.cb_dark.setChecked(False)
            self.controls.cb_dark.hide()
        except Exception:
            pass

    # ---------- IM file IO (mirrors baseline MainWindow) ----------
    def open_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open .im file(s)", "", "IM files (*.im);;All files (*)")
        if not paths: return
        try:
            self.sources = []
            for i, p in enumerate(paths):
                self._ingest_file(Path(p), i)
            self._refresh_after_ingest(status_prefix="Loaded")
        except Exception as e:
            log_ex(e); QMessageBox.critical(self, "Failed to load", f"{e}\n\nSee log: {LOGFILE}")

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add .im file(s)", "", "IM files (*.im);;All files (*)")
        if not paths: return
        try:
            start = len(self.sources)
            for k, p in enumerate(paths):
                self._ingest_file(Path(p), start + k)
            self._refresh_after_ingest(status_prefix="Added")
        except Exception as e:
            log_ex(e); QMessageBox.critical(self, "Failed to add files", f"{e}\n\nSee log: {LOGFILE}")

    def clear_all(self):
        self.sources = []
        self.series_panel.set_series([])
        self.plot_grid.clear_axes(); self.plot_grid.draw()
        self.statusBar().showMessage("Cleared")

    def _ingest_file(self, path: Path, index: int):
        parser = pick_parser(path)
        records = parser.parse(path)
        self.sources.append(DataSource(path, parser.name(), records, index))

    def _refresh_after_ingest(self, status_prefix="Loaded"):
        # Update left panel and apply current controls to IM plot grid
        self.series_panel.set_series(self.sources)
        self.statusBar().showMessage(f"{status_prefix} {len(self.sources)} file(s)")
        try:
            self.plot_grid.set_show_legends(self.controls.show_legends())
            self.plot_grid.set_hpad(self.controls.hpad())
            self.plot_grid.set_wpad(self.controls.wpad())
            self.plot_grid.set_show_grid(self.controls.show_grid())
            self.plot_grid.legend_mode = self.controls.legend_mode()
            self.plot_grid.legend_width = self.controls.legend_width()
            self.plot_grid.legend_fs = self.controls.legend_fontsize()
            self.plot_grid.legend_ncol = self.controls.legend_ncol()
            self.plot_grid.compact_labels = self.controls.compact_labels()
            self.plot_grid.set_axis_toggle(*self.controls.axis_toggles())
            self.plot_grid.set_axis_scales(*self.controls.axis_scales())
        except Exception:
            pass
        self.update_plots()

    def _refresh_tables(self):
        # Build mapping for the Tables tab from current selection (fallback to all)
        combined_map = {}
        combined_labels = []
        selected = []
        try:
            selected = self.series_panel.selected_series()
        except Exception:
            selected = []
        if selected:
            for (si, lab) in selected:
                if si < 0 or si >= len(self.sources): 
                    continue
                src = self.sources[si]
                recs = src.freq_to_records.get(lab, [])
                if not recs:
                    continue
                label = f"{src.file_label} • {lab}"
                combined_map[label] = recs
                combined_labels.append(label)
        else:
            for src in self.sources:
                for lab, recs in src.freq_to_records.items():
                    label = f"{src.file_label} • {lab}"
                    combined_map[label] = recs
                    combined_labels.append(label)
        try:
            self.tables_tab.set_data(combined_map, combined_labels, self.controls.curve_pref(), self.controls.ignore_a2())
        except Exception:
            pass

    # ---------- Plotting (same logic as baseline MainWindow) ----------
    def _build_mapping_for_plots(self, selected):
        mapping = {}; labels = []
        for si, lab in selected:
            if si < 0 or si >= len(self.sources): 
                continue
            src = self.sources[si]
            recs = src.freq_to_records.get(lab, [])
            if not recs: 
                continue
            label = f"{src.file_label} • {lab}"
            mapping[label] = (src.index, recs)
            labels.append(label)
        return mapping, labels

    def update_plots(self):
        if not self.sources:
            return
        selected = self.series_panel.selected_series()
        if not selected:
            self.plot_grid.clear_axes(); self.plot_grid.draw()
            self.statusBar().showMessage("No series selected")
            return
        curve_pref = self.controls.curve_pref()
        ignore_a2 = self.controls.ignore_a2()
        mapping, labels = self._build_mapping_for_plots(selected)

        # Detect ACPR on any selected series (mirrors baseline choice)
        show_acpr_any = False
        for label in labels:
            _, recs = mapping.get(label, (0, []))
            rec = PlotGrid._choose_record(recs, curve_pref)
            if rec is None:
                continue
            df = compute_metrics(rec, use_gamma_source=False, ignore_a2=ignore_a2)
            sL = df.get("ACPR Lower [dBc] @ f0"); sU = df.get("ACPR Upper [dBc] @ f0")
            if sL is not None and np.isfinite(sL.values).any(): 
                show_acpr_any = True; break
            if sU is not None and np.isfinite(sU.values).any(): 
                show_acpr_any = True; break

        try:
            self.plot_grid.overlay_frequency_curves(mapping, labels, curve_pref, ignore_a2, show_acpr_any)
            tail = "Pout=|B2|^2" if ignore_a2 else "Pout=|B2|^2−|A2|^2"
            self.statusBar().showMessage(f"Plotted {len(labels)} series • Pref: {curve_pref.upper()} • {tail}")
        except Exception as e:
            log_ex(e)
            QMessageBox.critical(self, "Plot error", f"{e}")

    # ---------- Controls sync & tab changes ----------
    def _on_tab_changed(self, idx: int):
        w = self.tabs.widget(idx)
        if w is self.sparam_tab:
            self.controls.set_mode('sparam')
        else:
            self.controls.set_mode('im')
        self.update_plots()

    def _on_controls_changed(self):
        try:
            self.plot_grid.apply_theme(
                dark=False,  # force light
                show_grid=self.controls.show_grid(),
                lw=self.controls.linewidth(),
                alpha=self.controls.alpha()
            )
        except Exception:
            pass
        try:
            self.plot_grid.set_show_legends(self.controls.show_legends())
            self.plot_grid.set_hpad(self.controls.hpad())
            self.plot_grid.set_wpad(self.controls.wpad())
            self.plot_grid.set_show_grid(self.controls.show_grid())
            self.plot_grid.legend_mode = self.controls.legend_mode()
            self.plot_grid.legend_width = self.controls.legend_width()
            self.plot_grid.legend_fs = self.controls.legend_fontsize()
            self.plot_grid.legend_ncol = self.controls.legend_ncol()
            self.plot_grid.compact_labels = self.controls.compact_labels()
            self.plot_grid.set_axis_toggle(*self.controls.axis_toggles())
            self.plot_grid.set_axis_scales(*self.controls.axis_scales())
        except Exception:
            pass
        self.update_plots()

# [__main__ removed for single-file rebuild]



# ---------- Mouse wheel forwarding ----------
from PySide6 import QtGui as _QtGui
class _WheelForwardFilter(QtCore.QObject):
    def __init__(self, target_scroll_area):
        super().__init__(); self.target = target_scroll_area
    def eventFilter(self, obj, event):
        if isinstance(event, _QtGui.QWheelEvent):
            QtWidgets.QApplication.sendEvent(self.target.viewport(), event); return True
        return False



if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = MainWindowSingle()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec())
