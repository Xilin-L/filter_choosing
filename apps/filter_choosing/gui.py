from __future__ import annotations

from core import materialPropertiesData as mpd

import tkinter as tk
from tkinter import ttk, messagebox

from .engine import (
    FILTER_OPTIONS,
    getRuleOfThumbFilterThickness,
    runAll,
)

# Minimal lists for v1. Replace/extend later.
MACHINES = ["L1", "L4", "ANU3", "RT1", "DB1", "DB2", "DB3", "ANU4"]

# Default modes; we'll adjust ANU4 when machine changes.
MODES_DEFAULT = ["S", "M", "L"]

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
    root.title("Filter Choosing Tool (Basic)")

    # -----------------------------
    # Variables
    # -----------------------------
    machineVar = tk.StringVar(value=MACHINES[0])
    modeVar = tk.StringVar(value="S")
    filterVar = tk.StringVar(value=FILTER_OPTIONS[0])
    defaultSample = "sandstone" if "sandstone" in SAMPLE_MATERIALS else SAMPLE_MATERIALS[0]
    sampleMaterialVar = tk.StringVar(value=defaultSample)

    kvpVar = tk.StringVar(value="120")
    sampleDiameterVar = tk.StringVar(value="10.0")

    # ROT output and main filter thickness
    rotThicknessVar = tk.StringVar(value="")
    filterThicknessVar = tk.StringVar(value="0.0")

    # Scatter/BH controls
    coneAngleVar = tk.StringVar(value="10.0")
    sampleDiameterVxVar = tk.StringVar(value="256")

    # Flux inputs
    sourceCurrentUaVar = tk.StringVar(value="60")   # allow blank later if you want
    powerWattVar = tk.StringVar(value="")           # optional
    exposureTimeVar = tk.StringVar(value="1.0")
    accumulationNumVar = tk.StringVar(value="1")

    # -----------------------------
    # Layout
    # -----------------------------
    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    row = 0

    def addLabelEntry(label: str, var: tk.StringVar):
        nonlocal row
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        e = ttk.Entry(frm, textvariable=var)
        e.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        return e

    def addLabelCombo(label: str, var: tk.StringVar, values: list[str]):
        nonlocal row
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        c = ttk.Combobox(frm, textvariable=var, values=values, state="readonly")
        c.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        return c

    machineCombo = addLabelCombo("Machine", machineVar, MACHINES)
    modeCombo = addLabelCombo("Mode", modeVar, MODES_DEFAULT)
    addLabelEntry("kVp", kvpVar)

    addLabelCombo("Filter material", filterVar, FILTER_OPTIONS)
    addLabelEntry("Filter thickness (mm)", filterThicknessVar)

    addLabelCombo("Sample material", sampleMaterialVar, SAMPLE_MATERIALS)
    addLabelEntry("Sample diameter (mm)", sampleDiameterVar)

    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    addLabelEntry("Cone angle (deg) [scatter]", coneAngleVar)
    addLabelEntry("Simulation diameter voxels [BH]", sampleDiameterVxVar)

    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    addLabelEntry("Source current (µA) [flux]", sourceCurrentUaVar)
    addLabelEntry("Power (W) [optional]", powerWattVar)
    addLabelEntry("Exposure time (s)", exposureTimeVar)
    addLabelEntry("Accumulation num", accumulationNumVar)

    ttk.Separator(frm).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    # ROT row (output + buttons)
    ttk.Label(frm, text="ROT thickness (mm)").grid(row=row, column=0, sticky="w", pady=2)
    rotEntry = ttk.Entry(frm, textvariable=rotThicknessVar, state="readonly")
    rotEntry.grid(row=row, column=1, sticky="ew", pady=2)
    row += 1

    btnRow = ttk.Frame(frm)
    btnRow.grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
    btnRow.columnconfigure(0, weight=1)
    btnRow.columnconfigure(1, weight=1)
    btnRow.columnconfigure(2, weight=1)

    computeRotBtn = ttk.Button(btnRow, text="Compute ROT")
    useRotBtn = ttk.Button(btnRow, text="Use ROT thickness")
    runAllBtn = ttk.Button(btnRow, text="Run All")

    computeRotBtn.grid(row=0, column=0, sticky="ew", padx=2)
    useRotBtn.grid(row=0, column=1, sticky="ew", padx=2)
    runAllBtn.grid(row=0, column=2, sticky="ew", padx=2)

    row += 1

    # Output text box
    ttk.Label(frm, text="Output").grid(row=row, column=0, sticky="nw", pady=(10, 2))
    outputTxt = tk.Text(frm, height=10, wrap="word")
    outputTxt.grid(row=row, column=1, sticky="nsew", pady=(10, 2))
    frm.rowconfigure(row, weight=1)
    row += 1

    def writeOutput(text: str) -> None:
        outputTxt.delete("1.0", "end")
        outputTxt.insert("1.0", text)

    # -----------------------------
    # Dynamic behavior
    # -----------------------------
    def updateModesForMachine(*_):
        m = machineVar.get()
        if m == "ANU4":
            modes = ["S", "L"]
            if modeVar.get() not in modes:
                modeVar.set("S")
        else:
            modes = ["S", "M", "L"]
            if modeVar.get() not in modes:
                modeVar.set("S")
        modeCombo["values"] = modes

    machineCombo.bind("<<ComboboxSelected>>", updateModesForMachine)
    updateModesForMachine()

    # -----------------------------
    # Button callbacks
    # -----------------------------
    def onComputeRot():
        try:
            kvp = _safeFloat(kvpVar.get().strip(), "kVp")
            sampleDiameterMm = _safeFloat(sampleDiameterVar.get().strip(), "Sample diameter (mm)")
            tRot = getRuleOfThumbFilterThickness(
                kvp=kvp,
                filterMaterial=filterVar.get(),
                sampleMaterial=sampleMaterialVar.get(),
                sampleDiameterMm=sampleDiameterMm,
            )
            rotThicknessVar.set(f"{tRot:.6g}")
            writeOutput(f"Rule-of-thumb thickness:\n  {tRot:.6g} mm\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def onUseRot():
        t = rotThicknessVar.get().strip()
        if not t:
            messagebox.showinfo("Info", "Compute ROT thickness first.")
            return
        filterThicknessVar.set(t)

    def onRunAll():
        try:
            machine = machineVar.get()
            mode = modeVar.get()

            kvp = _safeFloat(kvpVar.get().strip(), "kVp")
            filterThicknessMm = _safeFloat(filterThicknessVar.get().strip(), "Filter thickness (mm)")
            sampleDiameterMm = _safeFloat(sampleDiameterVar.get().strip(), "Sample diameter (mm)")

            coneAngleDeg = _safeFloat(coneAngleVar.get().strip(), "Cone angle (deg)")
            sampleDiameterVx = _safeInt(sampleDiameterVxVar.get().strip(), "Simulation diameter voxels")

            exposureTimeSec = _safeFloat(exposureTimeVar.get().strip(), "Exposure time (s)")
            accumulationNum = _safeInt(accumulationNumVar.get().strip(), "Accumulation num")

            # Allow blank for current/power
            currentStr = sourceCurrentUaVar.get().strip()
            powerStr = powerWattVar.get().strip()
            sourceCurrentUa = float(currentStr) if currentStr != "" else None
            powerWatt = float(powerStr) if powerStr != "" else None

            results = runAll(
                machine=machine,
                mode=mode,
                kvp=kvp,
                filterMaterial=filterVar.get(),
                filterThicknessMm=filterThicknessMm,
                sampleMaterial=sampleMaterialVar.get(),
                sampleDiameterMm=sampleDiameterMm,
                coneAngleDeg=coneAngleDeg,
                sampleDiameterVx=sampleDiameterVx,
                sourceCurrentUa=sourceCurrentUa,
                powerWatt=powerWatt,
                exposureTimeSec=exposureTimeSec,
                accumulationNum=accumulationNum,
            )

            # Pretty print
            lines = []
            lines.append("Inputs:")
            lines.append(f"  machine={machine}, mode={mode}")
            lines.append(f"  kvp={kvp}")
            lines.append(f"  filter={filterVar.get()}, thicknessMm={filterThicknessMm}")
            lines.append(f"  sampleMaterial={sampleMaterialVar.get()}, sampleDiameterMm={sampleDiameterMm}")
            lines.append(f"  coneAngleDeg={coneAngleDeg}, sampleDiameterVx={sampleDiameterVx}")
            lines.append(f"  sourceCurrentUa={sourceCurrentUa}, powerWatt={powerWatt}")
            lines.append(f"  exposureTimeSec={exposureTimeSec}, accumulationNum={accumulationNum}")
            lines.append("")
            lines.append("Outputs:")
            lines.append(f"  transmission percentage = {results['transmission percentage']:.4f}")
            lines.append(f"  effective attenuation per cm = {results['effective attenuation per cm']:.3f}")
            lines.append(f"  scatter/transmission in percentage = {results['scatter/transmission in percentage']:.4f}")
            lines.append(f"  bhc factor = {results['bhc factor']:.4f}")

            writeOutput("\n".join(lines))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    computeRotBtn.config(command=onComputeRot)
    useRotBtn.config(command=onUseRot)
    runAllBtn.config(command=onRunAll)

    root.minsize(700, 520)
    root.mainloop()
