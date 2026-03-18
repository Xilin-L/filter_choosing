from __future__ import annotations
from typing import Any
import math

import numpy as np

from core import filterPerformance as fp
from core import scatteringSimulation as scat
from core import beamHardeningSimulation as bhs
from core import xraySimulation as xs
from core import materialPropertiesData as mpd

try:
    import xraydb
except ImportError:
    xraydb = None

FILTER_OPTIONS = ["Al", "Fe", "Cu", "304sstl"]


def _buildEquivalentMaterialProperties(sampleComposition: dict[str, float]) -> tuple[
    list[float], list[str], float, dict]:
    """
    Build equivalent elemental composition + equivalent density for a multi-material mixture.
    """
    if not sampleComposition:
        raise ValueError("Sample composition cannot be empty.")

    # 1. Sum and validate the volume fractions (Strictly 1.0)
    total_vol = sum(sampleComposition.values())

    # Check if sum is close to 1.0 (tolerance of 0.01 for floating point math)
    if not math.isclose(total_vol, 1.0, abs_tol=0.01):
        raise ValueError(f"The sum of volume fractions must equal 1.0. "
                         f"Current sum is {total_vol:.4f}. Please check your inputs.")

    norm_vols = {mat: (v / total_vol) for mat, v in sampleComposition.items() if v > 0}

    if not norm_vols:
        raise ValueError("No materials with positive volume fractions provided.")

    # 2. Extract properties and calculate masses
    total_mass = 0.0
    material_data = {}

    for mat, v_frac in norm_vols.items():
        w_elem, s_elem, rho = mpd.getMaterialProperties(mat)
        w_elem = np.asarray(w_elem, dtype=float)

        if len(w_elem) != len(s_elem):
            raise ValueError(f"Material '{mat}' has mismatched symbols/weights.")
        if np.any(w_elem < 0):
            raise ValueError(f"Material '{mat}' has negative elemental weights.")
        if float(np.sum(w_elem)) <= 0:
            raise ValueError(f"Material '{mat}' has zero elemental weight sum.")

        w_elem = w_elem / float(np.sum(w_elem))

        rho = float(rho)
        mass = v_frac * rho
        total_mass += mass

        material_data[mat] = {"w_elem": w_elem, "s_elem": list(s_elem), "rho": rho, "v_frac": v_frac, "mass": mass}

    if total_mass <= 0:
        raise ValueError("Invalid mixture: total mass <= 0.")

    # 3. Merge elemental compositions based on mass fractions
    symbolToWeight: dict[str, float] = {}
    mixtureDensity = 0.0

    meta = {"isMultiMaterial": len(norm_vols) > 1, "composition": {}}

    for mat, data in material_data.items():
        massFrac = data["mass"] / total_mass
        mixtureDensity += data["v_frac"] * data["rho"]

        for sym, w in zip(data["s_elem"], data["w_elem"]):
            symbolToWeight[sym] = symbolToWeight.get(sym, 0.0) + (massFrac * float(w))

        meta["composition"][mat] = {"volumeFrac": data["v_frac"], "massFrac": massFrac, "density": data["rho"]}

    # 4. Stable ordering and final normalization
    try:
        elementSymbols = sorted(symbolToWeight.keys(), key=lambda x: xraydb.atomic_number(x))
    except Exception:
        elementSymbols = sorted(symbolToWeight.keys())

    elementWeights = np.asarray([symbolToWeight[s] for s in elementSymbols], dtype=float)

    wsum = float(np.sum(elementWeights))
    if wsum <= 0:
        raise ValueError("Mixture elemental weights sum to zero.")
    elementWeights = elementWeights / wsum

    return elementWeights.tolist(), elementSymbols, float(mixtureDensity), meta


def getRuleOfThumbFilterThickness(kvp: float, filterMaterial: str, sampleDiameterMm: float,
        sampleComposition: dict[str, float]) -> float:
    materialWeights, materialSymbols, density, _ = _buildEquivalentMaterialProperties(sampleComposition)

    return float(
        fp.getRuleOfThumbFilterThickness(kvp=kvp, filterMaterial=filterMaterial, sampleDiameterMm=sampleDiameterMm,
            materialWeights=materialWeights, materialSymbols=materialSymbols, density=density, ))


def runAll(kvp: float, filterMaterial: str, filterThicknessMm: float, sampleDiameterMm: float,
        sampleComposition: dict[str, float], coneAngleDeg: float = 10.0, sampleDiameterVx: int = 256,
        tubeThicknessMm: float = 0.0, tubeMaterial: str = "Al", ) -> dict:
    if filterMaterial not in FILTER_OPTIONS:
        raise ValueError(f"Unsupported filter material: {filterMaterial}")
    if filterThicknessMm <= 0:
        raise ValueError("Filter thickness must be > 0.")
    if sampleDiameterMm <= 0:
        raise ValueError("Sample diameter/thickness must be > 0.")
    if sampleDiameterVx <= 0:
        raise ValueError("Simulation diameter voxels must be > 0.")
    if tubeThicknessMm < 0:
        raise ValueError("Tube thickness cannot be negative.")
    if coneAngleDeg <= 0:
        raise ValueError("Cone angle must be > 0.")

    materialWeights, materialSymbols, density, materialMeta = _buildEquivalentMaterialProperties(sampleComposition)

    # 1) Spectrum at detector after selected filter
    energyKeV, spectrumDet = fp.setSpectrum(kvp=kvp, filterMaterial=filterMaterial,
        filterThicknessMm=filterThicknessMm, )

    # 1b) Optional tube as extra filtering layer
    if tubeThicknessMm > 0.0:
        tubeWeights, tubeSymbols, tubeDensity = mpd.getMaterialProperties(tubeMaterial)
        tubeTrans = xs.calcTransmission(energyKeV, tubeWeights, tubeSymbols, tubeDensity, tubeThicknessMm / 10.0,
            # mm -> cm
        )
        spectrumDet = spectrumDet * tubeTrans

    # 2) Scatter %
    scatterPercent = scat.estimateSampleScatterAsPercentOfTransmission(energyKeV=energyKeV, spectrum=spectrumDet,
        sampleDiameterMm=sampleDiameterMm, coneAngleDeg=coneAngleDeg, materialWeights=materialWeights,
        materialSymbols=materialSymbols, density=density, )

    # 3) Beam hardening factor
    bhcFactor, voxelSizeMm, *_ = bhs.simulateBH(sampleDiameterMm=sampleDiameterMm, sampleDiameterVx=sampleDiameterVx,
        kvp=kvp, filterMaterial=filterMaterial, filterThicknessMm=filterThicknessMm, materialWeights=materialWeights,
        materialSymbols=materialSymbols, density=density, plotCurve=False, plotIdeal=False, plotBH=False,
        verbose=False, )

    # 4) Transmission and effective attenuation
    totalTransmission = xs.calcSpecTrans(energyKeV, spectrumDet, materialWeights, materialSymbols, density,
        sampleDiameterMm, )
    muPerCm = -np.log(totalTransmission) / sampleDiameterMm * 10.0

    # Build dynamic output dictionary
    output = {"mode": "multi-material mixture" if materialMeta["isMultiMaterial"] else "single material", }

    # Add composition breakdown dynamically
    for mat, props in materialMeta["composition"].items():
        output[f"[{mat}] vol frac"] = round(float(props["volumeFrac"]), 4)
        output[f"[{mat}] mass frac"] = round(float(props["massFrac"]), 4)
        output[f"[{mat}] density"] = round(float(props["density"]), 4)

    # Add remainder of standard metrics
    output.update({"sample density (equivalent)": round(float(density), 4), "filter material": filterMaterial,
        "filter thickness mm": round(float(filterThicknessMm), 4),
        "tube material": tubeMaterial if tubeThicknessMm > 0 else "-",
        "tube thickness mm": round(float(tubeThicknessMm), 4),
        "scatter/transmission in percentage": round(float(scatterPercent), 4), "bhc factor": round(float(bhcFactor), 4),
        "transmission percentage": round(float(totalTransmission * 100.0), 4),
        "effective attenuation per cm": round(float(muPerCm), 4), })

    return output


def optimizeScanParameters(kvpList: list[float], filterMaterial: str, sampleDiameterMm: float,
        sampleComposition: dict[str, float], targetBhcMax: float = 1.1, targetTransMin: float = 0.05,
        targetTransMax: float = 0.35, priority: str = "transmission", tubeMaterial: str = "Al",
        tubeThicknessMm: float = 2.0) -> list[dict]:
    materialWeights, materialSymbols, density, _ = _buildEquivalentMaterialProperties(sampleComposition)

    valid_configurations = []

    if "Al" in filterMaterial:
        thicknessStepMm = 0.5
    elif any(mat in filterMaterial for mat in ["Fe", "Cu", "304sstl"]):
        thicknessStepMm = 0.1
    else:
        thicknessStepMm = 0.1

    for kvp in kvpList:
        energyKeV, rawSpectrumIn = xs.generateEmittedSpectrum(kvp, filterThicknessMm=0.0)

        if tubeThicknessMm > 0:
            spectrumIn = fp.getFilteredSpectrum(energyKeV, rawSpectrumIn, tubeMaterial, tubeThicknessMm)
        else:
            spectrumIn = rawSpectrumIn

        sampleAttPerCm = xs.getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, density)

        configs_for_kvp = []

        rot_t = getRuleOfThumbFilterThickness(kvp, filterMaterial, sampleDiameterMm, sampleComposition)
        start_t = max(0.0, 0.5 * rot_t)
        end_t = 1.5 * rot_t

        for t_mm in np.arange(start_t, end_t + thicknessStepMm, thicknessStepMm):
            spectrumFilt = fp.getFilteredSpectrum(energyKeV, spectrumIn, filterMaterial, t_mm)
            spectrumDet = xs.detectedSpectrum(energyKeV, spectrumFilt)

            if np.sum(spectrumDet) <= 0:
                continue

            totalTransmission = xs.calcSpecTrans(energyKeV, spectrumDet, materialWeights, materialSymbols, density,
                                                 sampleDiameterMm)

            if targetTransMin <= totalTransmission <= targetTransMax:
                fluxFraction = np.sum(spectrumFilt) / np.sum(rawSpectrumIn)
                A, n = bhs.estimateBeamHardening(spectrumDet, sampleAttPerCm, sampleDiameterMm, plot=False)
                bhcFactor = 1.0 / n

                configs_for_kvp.append({"kvp": float(kvp), "filterThicknessMm": round(float(t_mm), 2),
                    "transmission": round(float(totalTransmission * 100), 2),
                    "flux": round(float(fluxFraction * 100), 4), "bhcFactor": round(float(bhcFactor), 4),
                    "meetsBhcTarget": bhcFactor <= targetBhcMax})

        if not configs_for_kvp:
            continue

        if priority == "bhc":
            best_for_kvp = min(configs_for_kvp, key=lambda x: x["bhcFactor"])
        else:
            best_for_kvp = max(configs_for_kvp, key=lambda x: x["transmission"] * x["flux"])

        if best_for_kvp:
            valid_configurations.append(best_for_kvp)

    return valid_configurations