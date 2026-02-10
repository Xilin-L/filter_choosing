from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from core import materialPropertiesData as mpd
from .engine import (
    FILTER_OPTIONS,
    getRuleOfThumbFilterThickness,
    runAll,
)

SAMPLE_MATERIALS = sorted(
    set(mpd.materialProperties.keys()) | set(mpd.materialNameMapping.keys())
)


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
    volFrac2Var = tk.StringVar(value="0.10")        # volume fraction of material2 [0,1]

    # Tube controls (extra filtering)
    tubeEnabledVar = tk.BooleanVar(value=True)
    tubeMaterialVar = tk.StringVar(value="Al")
    tubeThicknessVar = tk.StringVar(value="2.0")    # mm

    # ROT + chosen filter thickness
    rotThicknessVar = tk.StringVar(value="")
    filterThicknessVar = tk.StringVar(value="0.0")

    # Scatter/BH controls
    coneAngleVar = tk.StringVar(value="10.0")
    sampleDiameterVxVar = tk.StringVar(value="256")

    # -----------------------------
    # Layout
    # -----------------------------
    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    row = 0

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

    # Base inputs
    addLabelEntry("kVp", kvpVar)
    addLabelCombo("Filter material", filterVar, FILTER_OPTIONS)
    addLabelEntry("Filter thickness (mm)", filterThicknessVar)

    addLabelCombo("Material 1", material1Var, SAMPLE_MATERIALS)

    useMaterial2Check = ttk.Checkbutton(
        frm,
        text="Enable material 2 (mixture mode)",
        variable=useMaterial2Var,
    )
    useMaterial2Check.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
    row += 1

    material2Combo = addLabelCombo("Material 2", material2Var, SAMPLE_MATERIALS, state="disabled")
    volFrac2Entry = addLabelEntry("Volume fraction of material 2 (0 to 1)", volFrac2Var, state="disabled")

    addLabelEntry("Sample diameter/thickness (mm)", sampleDiameterVar)

    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    # Tube section
    tubeEnabledCheck = ttk.Checkbutton(
        frm,
        text="Enable tube as extra filtering",
        variable=tubeEnabledVar,
    )
    tubeEnabledCheck.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
    row += 1

    tubeMaterialCombo = addLabelCombo("Tube material", tubeMaterialVar, ["Al"], state="readonly")
    tubeThicknessEntry = addLabelEntry("Tube thickness (mm)", tubeThicknessVar)

    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    # Advanced controls
    addLabelEntry("Cone angle (deg)", coneAngleVar)
    addLabelEntry("Simulation diameter voxels", sampleDiameterVxVar)

    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    # ROT row
    ttk.Label(frm, text="ROT thickness (mm)").grid(row=row, column=0, sticky="w", pady=2)
    rotEntry = ttk.Entry(frm, textvariable=rotThicknessVar, state="readonly")
    rotEntry.grid(row=row, column=1, sticky="ew", pady=2)
    row += 1

    # Buttons
    btnRow = ttk.Frame(frm)
    btnRow.grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
    btnRow.columnconfigure(0, weight=1)
    btnRow.columnconfigure(1, weight=1)
    btnRow.columnconfigure(2, weight=1)

    computeRotBtn = ttk.Button(btnRow, text="Compute ROT")
    useRotBtn = ttk.Button(btnRow, text="Use ROT thickness")
    runBtn = ttk.Button(btnRow, text="Run")

    computeRotBtn.grid(row=0, column=0, sticky="ew", padx=2)
    useRotBtn.grid(row=0, column=1, sticky="ew", padx=2)
    runBtn.grid(row=0, column=2, sticky="ew", padx=2)
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

        return {
            "kvp": kvp,
            "filterMaterial": filterMaterial,
            "sampleDiameterMm": sampleDiameterMm,
            "material1": material1,
            "material2": material2,
            "volumeFractionMaterial2": volFrac2,
            "tubeMaterial": tubeMaterial,
            "tubeThicknessMm": tubeThicknessMm,
            "coneAngleDeg": coneAngleDeg,
            "sampleDiameterVx": sampleDiameterVx,
        }

    def onComputeRot():
        try:
            p = _readInputs()
            tRot = getRuleOfThumbFilterThickness(
                kvp=p["kvp"],
                filterMaterial=p["filterMaterial"],
                sampleDiameterMm=p["sampleDiameterMm"],
                material1=p["material1"],
                material2=p["material2"],
                volumeFractionMaterial2=p["volumeFractionMaterial2"],
            )
            rotThicknessVar.set(f"{tRot:.4f}")
            writeOutput(f"Rule-of-thumb thickness:\n  {tRot:.4f} mm\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def onUseRot():
        t = rotThicknessVar.get().strip()
        if not t:
            messagebox.showinfo("Info", "Compute ROT thickness first.")
            return
        filterThicknessVar.set(t)

    def onRun():
        try:
            p = _readInputs()
            filterThicknessMm = _safeFloat(filterThicknessVar.get().strip(), "Filter thickness (mm)")

            results = runAll(
                kvp=p["kvp"],
                filterMaterial=p["filterMaterial"],
                filterThicknessMm=filterThicknessMm,
                sampleDiameterMm=p["sampleDiameterMm"],
                material1=p["material1"],
                material2=p["material2"],
                volumeFractionMaterial2=p["volumeFractionMaterial2"],
                coneAngleDeg=p["coneAngleDeg"],
                sampleDiameterVx=p["sampleDiameterVx"],
                tubeThicknessMm=p["tubeThicknessMm"],
                tubeMaterial=p["tubeMaterial"],
            )

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

    computeRotBtn.config(command=onComputeRot)
    useRotBtn.config(command=onUseRot)
    runBtn.config(command=onRun)

    useMaterial2Var.trace_add("write", updateUiState)
    tubeEnabledVar.trace_add("write", updateUiState)
    updateUiState()

    root.minsize(800, 620)
    root.mainloop()
