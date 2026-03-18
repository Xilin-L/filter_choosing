from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import os
import re
import sys

from core import materialPropertiesData as mpd
from .engine import (FILTER_OPTIONS, getRuleOfThumbFilterThickness, runAll, optimizeScanParameters)

SAMPLE_MATERIALS = sorted(set(mpd.materialProperties.keys()) | set(mpd.materialNameMapping.keys()))


def _safeFloat(s: str, fieldName: str) -> float:
    try:
        return float(s)
    except Exception:
        raise ValueError(f"{fieldName} must be a number.")


def _safeInt(s: str, fieldName: str) -> int:
    try:
        return int(s)
    except Exception:
        raise ValueError(f"{fieldName} must be an integer.")


def _get_app_version() -> str:
    """Reads the version dynamically from pyproject.toml."""
    try:
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(current_dir))

        toml_path = os.path.join(base_dir, "pyproject.toml")

        if os.path.exists(toml_path):
            with open(toml_path, "r", encoding="utf-8") as f:
                content = f.read()
                match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return " not found"


def main() -> None:
    root = tk.Tk()
    app_version = _get_app_version()
    root.title(f"Filter Choosing Tool v{app_version}")

    # -----------------------------
    # Variables
    # -----------------------------
    filterVar = tk.StringVar(value=FILTER_OPTIONS[0])
    kvpVar = tk.StringVar(value="120")
    sampleDiameterVar = tk.StringVar(value="10.0")  # mm

    # Tube controls (extra filtering)
    tubeEnabledVar = tk.BooleanVar(value=False)
    tubeMaterialVar = tk.StringVar(value="Al")
    tubeThicknessVar = tk.StringVar(value="2.0")  # mm

    # Chosen filter thickness
    filterThicknessVar = tk.StringVar(value="0.0")

    # Scatter/BH controls
    coneAngleVar = tk.StringVar(value="10.0")
    sampleDiameterVxVar = tk.StringVar(value="256")

    # Priority for optimization
    priorityVar = tk.StringVar(value="transmission")

    # -----------------------------
    # Layout
    # -----------------------------
    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    # -----------------------------------------------------------------
    # TOP SECTION: Split into Core Inputs (Left) and ROT Tools (Right)
    # -----------------------------------------------------------------
    topSplit = ttk.Frame(frm)
    topSplit.grid(row=0, column=0, columnspan=2, sticky="nsew")
    topSplit.columnconfigure(0, weight=1)
    topSplit.columnconfigure(1, weight=1)

    # -- LEFT: Core Inputs --
    leftCoreFrm = ttk.LabelFrame(topSplit, text="Core Parameters", padding=10)
    leftCoreFrm.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

    coreRow = 0

    def addCoreEntry(label: str, var: tk.StringVar, state: str = "normal"):
        nonlocal coreRow
        ttk.Label(leftCoreFrm, text=label).grid(row=coreRow, column=0, sticky="w", pady=2)
        e = ttk.Entry(leftCoreFrm, textvariable=var, state=state)
        e.grid(row=coreRow, column=1, sticky="ew", pady=2, padx=5)
        coreRow += 1
        return e

    def addCoreCombo(label: str, var: tk.StringVar, values: list[str], state: str = "readonly"):
        nonlocal coreRow
        ttk.Label(leftCoreFrm, text=label).grid(row=coreRow, column=0, sticky="w", pady=2)
        c = ttk.Combobox(leftCoreFrm, textvariable=var, values=values, state=state)
        c.grid(row=coreRow, column=1, sticky="ew", pady=2, padx=5)
        coreRow += 1
        return c

    addCoreEntry("kVp", kvpVar)
    addCoreCombo("Filter material", filterVar, FILTER_OPTIONS)
    addCoreEntry("Filter thickness (mm)", filterThicknessVar)
    addCoreEntry("Sample diameter/thickness (mm)", sampleDiameterVar)

    # --- DYNAMIC SAMPLE COMPOSITION SECTION ---
    ttk.Separator(leftCoreFrm).grid(row=coreRow, column=0, columnspan=2, sticky="ew", pady=10)
    coreRow += 1

    # Updated Label Text
    ttk.Label(leftCoreFrm, text="Sample Composition (Fractions sum to 1):").grid(row=coreRow, column=0, columnspan=2,
                                                                                 sticky="w")
    coreRow += 1

    compositionFrm = ttk.Frame(leftCoreFrm)
    compositionFrm.grid(row=coreRow, column=0, columnspan=2, sticky="ew", pady=5)
    coreRow += 1

    headerFrm = ttk.Frame(compositionFrm)
    headerFrm.pack(fill="x", pady=(0, 2))
    ttk.Label(headerFrm, text="Material", width=19).pack(side="left", padx=(0, 5))
    ttk.Label(headerFrm, text="Fraction", width=10).pack(side="left")

    rowsFrm = ttk.Frame(compositionFrm)
    rowsFrm.pack(fill="x")

    material_rows = []

    def add_material_row(default_mat=SAMPLE_MATERIALS[0], default_frac="1.0"):
        row_frame = ttk.Frame(rowsFrm)
        row_frame.pack(fill="x", pady=2)

        mat_var = tk.StringVar(value=default_mat)
        frac_var = tk.StringVar(value=default_frac)

        cb = ttk.Combobox(row_frame, textvariable=mat_var, values=SAMPLE_MATERIALS, state="readonly", width=16)
        cb.pack(side="left", padx=(0, 5))

        ent = ttk.Entry(row_frame, textvariable=frac_var, width=10)
        ent.pack(side="left", padx=(0, 5))

        row_info = {"frame": row_frame, "mat_var": mat_var, "frac_var": frac_var}

        def remove_row():
            row_frame.destroy()
            material_rows.remove(row_info)

        btn = ttk.Button(row_frame, text="X", width=2, command=remove_row)
        if len(material_rows) > 0:
            btn.pack(side="left")

        material_rows.append(row_info)

    # Updated default to "1.0"
    add_material_row("cu" if "cu" in SAMPLE_MATERIALS else SAMPLE_MATERIALS[0], "1.0")

    # Updated default to "0.1" for newly added rows
    addMatBtn = ttk.Button(leftCoreFrm, text="+ Add Material",
                           command=lambda: add_material_row(SAMPLE_MATERIALS[0], "0.1"))
    addMatBtn.grid(row=coreRow, column=0, columnspan=2, sticky="w", pady=(0, 5))
    coreRow += 1

    # -- RIGHT: Rule of Thumb (ROT) --
    rightRotFrm = ttk.LabelFrame(topSplit, text="Rule of Thumb (ROT) Estimator", padding=10)
    rightRotFrm.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

    rotOutputLabel = ttk.Label(rightRotFrm, text="Calculated ROT: -- mm", font=("", 10, "bold"), foreground="blue")
    rotOutputLabel.pack(pady=(20, 15))

    computeRotBtn = ttk.Button(rightRotFrm, text="Compute ROT")
    computeRotBtn.pack(fill="x", padx=20, pady=5)

    useRotBtn = ttk.Button(rightRotFrm, text="Use ROT thickness")
    useRotBtn.pack(fill="x", padx=20, pady=5)

    # -----------------------------------------------------------------
    # BOTTOM SECTION: Tube Settings
    # -----------------------------------------------------------------
    row = 1
    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1
    ttk.Label(frm, text="Tube Settings").grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
    row += 1

    def addLabelEntry(label: str, var: tk.StringVar, state: str = "normal"):
        nonlocal row
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        e = ttk.Entry(frm, textvariable=var, state=state)
        e.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        return e

    def addLabelCombo(label: str, var: tk.StringVar, values: list[str], state: str = "readonly"):
        nonlocal row
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        c = ttk.Combobox(frm, textvariable=var, values=values, state=state)
        c.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        return c

    tubeEnabledCheck = ttk.Checkbutton(frm, text="Enable tube as extra filtering", variable=tubeEnabledVar)
    tubeEnabledCheck.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
    row += 1

    tubeMaterialCombo = addLabelCombo("Tube material", tubeMaterialVar, ["Al"], state="readonly")
    tubeThicknessEntry = addLabelEntry("Tube thickness (mm)", tubeThicknessVar)

    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    # -----------------------------------------------------------------
    # BOTTOM SECTION: Action Buttons & Optimization Settings
    # -----------------------------------------------------------------
    actionFrm = ttk.Frame(frm)
    actionFrm.grid(row=row, column=0, columnspan=2, sticky="ew", pady=15)
    actionFrm.columnconfigure(0, weight=1)
    actionFrm.columnconfigure(1, weight=1)

    runGroup = ttk.LabelFrame(actionFrm, text="Single Configuration", padding=10)
    runGroup.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    runGroup.columnconfigure(1, weight=1)

    ttk.Label(runGroup, text="Cone angle (deg)").grid(row=0, column=0, sticky="w", pady=2, padx=(0, 5))
    ttk.Entry(runGroup, textvariable=coneAngleVar).grid(row=0, column=1, sticky="ew", pady=2)

    ttk.Label(runGroup, text="Simulation diam (voxels)").grid(row=1, column=0, sticky="w", pady=2, padx=(0, 5))
    ttk.Entry(runGroup, textvariable=sampleDiameterVxVar).grid(row=1, column=1, sticky="ew", pady=2)

    runBtn = ttk.Button(runGroup, text="Run Full Simulation")
    runBtn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    optGroup = ttk.LabelFrame(actionFrm, text="Optimization Sweep", padding=10)
    optGroup.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    optGroup.columnconfigure(1, weight=1)

    ttk.Label(optGroup, text="Priority:").grid(row=0, column=0, sticky="w", pady=2)
    priFrame = ttk.Frame(optGroup)
    priFrame.grid(row=0, column=1, sticky="w", pady=2)
    ttk.Radiobutton(priFrame, text="Transmission", variable=priorityVar, value="transmission").pack(side="left", padx=(0, 10))
    ttk.Radiobutton(priFrame, text="BHC", variable=priorityVar, value="bhc").pack(side="left")

    optimizeBtn = ttk.Button(optGroup, text="Optimize Sweep")
    optimizeBtn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    row += 1

    # Output
    ttk.Label(frm, text="Output").grid(row=row, column=0, sticky="nw", pady=(10, 2))
    outputTxt = tk.Text(frm, height=14, wrap="word")
    outputTxt.grid(row=row, column=1, sticky="nsew", pady=(10, 2))
    frm.rowconfigure(row, weight=1)

    def writeOutput(text: str) -> None:
        outputTxt.delete("1.0", "end")
        outputTxt.insert("1.0", text)

    # -----------------------------
    # Dynamic UI state
    # -----------------------------
    def updateUiState(*_):
        tubeOn = tubeEnabledVar.get()
        tubeMaterialCombo.configure(state="readonly" if tubeOn else "disabled")
        tubeThicknessEntry.configure(state="normal" if tubeOn else "disabled")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def _readInputs():
        kvp = _safeFloat(kvpVar.get().strip(), "kVp")
        filterMaterial = filterVar.get()
        sampleDiameterMm = _safeFloat(sampleDiameterVar.get().strip(), "Sample diameter/thickness (mm)")

        # Parse the dynamic sample composition dictionary
        sampleComposition = {}
        for row_info in material_rows:
            mat = row_info["mat_var"].get()
            frac = _safeFloat(row_info["frac_var"].get().strip(), f"Amount for {mat}")
            # If they select the same material twice, add the fractions together safely
            sampleComposition[mat] = sampleComposition.get(mat, 0.0) + frac

        tubeOn = tubeEnabledVar.get()
        tubeMaterial = tubeMaterialVar.get()
        tubeThicknessMm = _safeFloat(tubeThicknessVar.get().strip(), "Tube thickness (mm)") if tubeOn else 0.0

        coneAngleDeg = _safeFloat(coneAngleVar.get().strip(), "Cone angle (deg)")
        sampleDiameterVx = _safeInt(sampleDiameterVxVar.get().strip(), "Simulation diameter voxels")

        return {
            "kvp": kvp,
            "filterMaterial": filterMaterial,
            "sampleDiameterMm": sampleDiameterMm,
            "sampleComposition": sampleComposition,
            "tubeMaterial": tubeMaterial,
            "tubeThicknessMm": tubeThicknessMm,
            "coneAngleDeg": coneAngleDeg,
            "sampleDiameterVx": sampleDiameterVx
        }

    lastComputedRot = [None]

    def onComputeRot():
        try:
            p = _readInputs()
            tRot = getRuleOfThumbFilterThickness(
                kvp=p["kvp"],
                filterMaterial=p["filterMaterial"],
                sampleDiameterMm=p["sampleDiameterMm"],
                sampleComposition=p["sampleComposition"]
            )
            lastComputedRot[0] = tRot
            rotOutputLabel.config(text=f"Calculated ROT: {tRot:.2f} mm")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def onUseRot():
        if lastComputedRot[0] is None:
            messagebox.showinfo("Info", "Compute ROT thickness first.")
            return
        filterThicknessVar.set(f"{lastComputedRot[0]:.4f}")

    def onRun():
        try:
            p = _readInputs()
            filterThicknessMm = _safeFloat(filterThicknessVar.get().strip(), "Filter thickness (mm)")

            results = runAll(
                kvp=p["kvp"],
                filterMaterial=p["filterMaterial"],
                filterThicknessMm=filterThicknessMm,
                sampleDiameterMm=p["sampleDiameterMm"],
                sampleComposition=p["sampleComposition"],
                coneAngleDeg=p["coneAngleDeg"],
                sampleDiameterVx=p["sampleDiameterVx"],
                tubeThicknessMm=p["tubeThicknessMm"],
                tubeMaterial=p["tubeMaterial"]
            )

            lines = []
            lines.append("Inputs:")
            lines.append(f"  kvp = {p['kvp']:.4f}")
            lines.append(f"  filter = {p['filterMaterial']}, thickness mm = {filterThicknessMm:.4f}")
            lines.append("  sample composition:")
            for mat, frac in p["sampleComposition"].items():
                lines.append(f"    - {mat}: {frac}")
            lines.append(f"  sample diameter/thickness mm = {p['sampleDiameterMm']:.4f}")
            lines.append(f"  tube material = {p['tubeMaterial'] if p['tubeThicknessMm'] > 0 else '-'}")
            lines.append(f"  tube thickness mm = {p['tubeThicknessMm']:.4f}")
            lines.append(f"  cone angle deg = {p['coneAngleDeg']:.4f}")
            lines.append(f"  sample diameter vx = {p['sampleDiameterVx']}")
            lines.append("")
            lines.append("Outputs:")

            for k, v in results.items():
                lines.append(f"  {k} = {v}")

            writeOutput("\n".join(lines))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def onOptimize():
        try:
            p = _readInputs()
            import numpy as np

            base_kvp = p["kvp"]
            kvp_min = int(base_kvp * 0.5)
            kvp_max = int(base_kvp * 1.5)
            kvpList = np.arange(kvp_min, kvp_max + 1, 10).tolist()

            priority = priorityVar.get()

            writeOutput(f"Running optimization sweep (kVp {kvp_min}-{kvp_max})...\n"
                        f"Priority: {priority.capitalize()}\nThis may take a moment.")
            root.update()

            targetBhcMax = 1.1
            targetTransMin = 0.05
            targetTransMax = 0.35

            results = optimizeScanParameters(
                kvpList=kvpList,
                filterMaterial=p["filterMaterial"],
                sampleDiameterMm=p["sampleDiameterMm"],
                sampleComposition=p["sampleComposition"],
                targetBhcMax=targetBhcMax,
                targetTransMin=targetTransMin,
                targetTransMax=targetTransMax,
                priority=priority,
                tubeMaterial=p["tubeMaterial"],
                tubeThicknessMm=p["tubeThicknessMm"]
            )

            if not results:
                writeOutput(f"No valid configurations found.\n"
                            f"Could not achieve TRANSMISSION between {targetTransMin * 100}% and {targetTransMax * 100}% "
                            f"around the ROT thickness for the selected kVp range.\n"
                            f"Try a different filter material or adjust your inputs.")
                return

            lines = [f"Optimization Results ({priority.upper()} Priority):", "-" * 85]
            for res in results:
                note = "" if res["meetsBhcTarget"] else "  <-- BHC > Target"
                lines.append(f"kVp: {res['kvp']:<4.0f} | Filter: {res['filterThicknessMm']:<4.1f} mm "
                             f"| Flux: {res['flux']:>6.2f}% | Trans: {res['transmission']:>5.2f}% "
                             f"| BHC: {res['bhcFactor']:.4f}{note}")

            writeOutput("\n".join(lines))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    computeRotBtn.config(command=onComputeRot)
    useRotBtn.config(command=onUseRot)
    runBtn.config(command=onRun)
    optimizeBtn.config(command=onOptimize)

    tubeEnabledVar.trace_add("write", updateUiState)
    updateUiState()

    root.minsize(800, 620)
    root.mainloop()

if __name__ == "__main__":
    main()