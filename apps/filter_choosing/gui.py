from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

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


def main() -> None:
    root = tk.Tk()
    root.title("Filter Choosing Tool")

    # -----------------------------
    # Variables
    # -----------------------------
    filterVar = tk.StringVar(value=FILTER_OPTIONS[0])
    defaultMat1 = "sandstone" if "sandstone" in SAMPLE_MATERIALS else SAMPLE_MATERIALS[0]
    defaultMat2 = "air" if "air" in SAMPLE_MATERIALS else SAMPLE_MATERIALS[0]

    material1Var = tk.StringVar(value=defaultMat1)
    material2Var = tk.StringVar(value=defaultMat2)

    useMaterial2Var = tk.BooleanVar(value=False)

    kvpVar = tk.StringVar(value="120")
    sampleDiameterVar = tk.StringVar(value="10.0")  # mm

    # Mixture controls
    volFrac2Var = tk.StringVar(value="0.10")  # volume fraction of material2 [0,1]

    # Tube controls (extra filtering)
    tubeEnabledVar = tk.BooleanVar(value=True)
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

    addCoreCombo("Material 1", material1Var, SAMPLE_MATERIALS)

    useMaterial2Check = ttk.Checkbutton(leftCoreFrm, text="Enable material 2 (mixture mode)",
        variable=useMaterial2Var, )
    useMaterial2Check.grid(row=coreRow, column=0, columnspan=2, sticky="w", pady=2)
    coreRow += 1

    material2Combo = addCoreCombo("Material 2", material2Var, SAMPLE_MATERIALS, state="disabled")
    volFrac2Entry = addCoreEntry("Volume fraction of material 2 (0 to 1)", volFrac2Var, state="disabled")

    addCoreEntry("Sample diameter/thickness (mm)", sampleDiameterVar)

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
    # BOTTOM SECTION: Advanced Settings & Actions
    # -----------------------------------------------------------------
    row = 1
    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1
    ttk.Label(frm, text="Advanced & Tube Settings").grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
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

    # Tube section
    tubeEnabledCheck = ttk.Checkbutton(frm, text="Enable tube as extra filtering", variable=tubeEnabledVar, )
    tubeEnabledCheck.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
    row += 1

    tubeMaterialCombo = addLabelCombo("Tube material", tubeMaterialVar, ["Al"], state="readonly")
    tubeThicknessEntry = addLabelEntry("Tube thickness (mm)", tubeThicknessVar)

    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    # Advanced controls
    addLabelEntry("Cone angle (deg)", coneAngleVar)
    addLabelEntry("Simulation diameter voxels", sampleDiameterVxVar)

    ttk.Label(frm, text="Optimize For:").grid(row=row, column=0, sticky="w", pady=2)
    priFrame = ttk.Frame(frm)
    priFrame.grid(row=row, column=1, sticky="w")
    ttk.Radiobutton(priFrame, text="Transmission", variable=priorityVar, value="transmission").pack(side="left",
                                                                                                    padx=(0, 10))
    ttk.Radiobutton(priFrame, text="BHC (Beam Quality)", variable=priorityVar, value="bhc").pack(side="left")
    row += 1

    # Action Buttons (Bottom)
    btnRow = ttk.Frame(frm)
    btnRow.grid(row=row, column=0, columnspan=2, sticky="ew", pady=15)
    btnRow.columnconfigure(0, weight=1)
    btnRow.columnconfigure(1, weight=1)

    runBtn = ttk.Button(btnRow, text="Run Full Simulation")
    optimizeBtn = ttk.Button(btnRow, text="Optimize Sweep")

    runBtn.grid(row=0, column=0, sticky="ew", padx=2)
    optimizeBtn.grid(row=0, column=1, sticky="ew", padx=2)
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
        useMat2 = useMaterial2Var.get()
        tubeOn = tubeEnabledVar.get()

        material2Combo.configure(state="readonly" if useMat2 else "disabled")
        volFrac2Entry.configure(state="normal" if useMat2 else "disabled")

        tubeMaterialCombo.configure(state="readonly" if tubeOn else "disabled")
        tubeThicknessEntry.configure(state="normal" if tubeOn else "disabled")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def _readInputs():
        kvp = _safeFloat(kvpVar.get().strip(), "kVp")
        filterMaterial = filterVar.get()
        sampleDiameterMm = _safeFloat(sampleDiameterVar.get().strip(), "Sample diameter/thickness (mm)")

        useMat2 = useMaterial2Var.get()
        material1 = material1Var.get()
        material2 = material2Var.get() if useMat2 else None
        volFrac2 = _safeFloat(volFrac2Var.get().strip(), "Volume fraction of material 2") if useMat2 else 0.0

        if not (0.0 <= volFrac2 <= 1.0):
            raise ValueError("Volume fraction of material 2 must be in [0, 1].")

        tubeOn = tubeEnabledVar.get()
        tubeMaterial = tubeMaterialVar.get()
        tubeThicknessMm = _safeFloat(tubeThicknessVar.get().strip(), "Tube thickness (mm)") if tubeOn else 0.0

        coneAngleDeg = _safeFloat(coneAngleVar.get().strip(), "Cone angle (deg)")
        sampleDiameterVx = _safeInt(sampleDiameterVxVar.get().strip(), "Simulation diameter voxels")

        return {"kvp": kvp, "filterMaterial": filterMaterial, "sampleDiameterMm": sampleDiameterMm,
            "material1": material1, "material2": material2, "volumeFractionMaterial2": volFrac2,
            "tubeMaterial": tubeMaterial, "tubeThicknessMm": tubeThicknessMm, "coneAngleDeg": coneAngleDeg,
            "sampleDiameterVx": sampleDiameterVx, }

    lastComputedRot = [None]

    def onComputeRot():
        try:
            p = _readInputs()
            tRot = getRuleOfThumbFilterThickness(kvp=p["kvp"], filterMaterial=p["filterMaterial"],
                sampleDiameterMm=p["sampleDiameterMm"], material1=p["material1"], material2=p["material2"],
                volumeFractionMaterial2=p["volumeFractionMaterial2"], )
            lastComputedRot[0] = tRot
            # Update the label in the right frame
            rotOutputLabel.config(text=f"Calculated ROT: {tRot:.2f} mm")

            # (The writeOutput line has been removed so it doesn't print to the bottom window)
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

            results = runAll(kvp=p["kvp"], filterMaterial=p["filterMaterial"], filterThicknessMm=filterThicknessMm,
                sampleDiameterMm=p["sampleDiameterMm"], material1=p["material1"], material2=p["material2"],
                volumeFractionMaterial2=p["volumeFractionMaterial2"], coneAngleDeg=p["coneAngleDeg"],
                sampleDiameterVx=p["sampleDiameterVx"], tubeThicknessMm=p["tubeThicknessMm"],
                tubeMaterial=p["tubeMaterial"], )

            lines = []
            lines.append("Inputs:")
            lines.append(f"  kvp = {p['kvp']:.4f}")
            lines.append(f"  filter = {p['filterMaterial']}, thickness mm = {filterThicknessMm:.4f}")
            lines.append(f"  material 1 = {p['material1']}")
            lines.append(f"  material 2 = {p['material2'] if p['material2'] else '-'}")
            lines.append(f"  vol frac material 2 = {p['volumeFractionMaterial2']:.4f}")
            lines.append(f"  sample diameter/thickness mm = {p['sampleDiameterMm']:.4f}")
            lines.append(f"  tube material = {p['tubeMaterial'] if p['tubeThicknessMm'] > 0 else '-'}")
            lines.append(f"  tube thickness mm = {p['tubeThicknessMm']:.4f}")
            lines.append(f"  cone angle deg = {p['coneAngleDeg']:.4f}")
            lines.append(f"  sample diameter vx = {p['sampleDiameterVx']}")
            lines.append("")
            lines.append("Outputs:")

            # print all outputs from engine (already rounded in engine)
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

            results = optimizeScanParameters(kvpList=kvpList, filterMaterial=p["filterMaterial"],
                sampleDiameterMm=p["sampleDiameterMm"], material1=p["material1"], material2=p["material2"],
                volumeFractionMaterial2=p["volumeFractionMaterial2"], targetBhcMax=targetBhcMax,
                targetTransMin=targetTransMin,  # <-- Passed variable
                targetTransMax=targetTransMax,  # <-- Passed variable
                priority=priority, tubeMaterial=p["tubeMaterial"], tubeThicknessMm=p["tubeThicknessMm"])

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

    useMaterial2Var.trace_add("write", updateUiState)
    tubeEnabledVar.trace_add("write", updateUiState)
    updateUiState()

    root.minsize(800, 620)
    root.mainloop()