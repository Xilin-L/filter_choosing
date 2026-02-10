from __future__ import annotations
from typing import Any

import numpy as np

from core import filterPerformance as fp
from core import scatteringSimulation as scat
from core import beamHardeningSimulation as bhs
from core import xraySimulation as xs
from core import materialPropertiesData as mpd


FILTER_OPTIONS = ["Al", "Fe", "Cu", "304sstl"]


def toSpekpyFilterMaterial(filterCode: str) -> str:
    # SpekPy has 302; your lab option is 304sstl and you consider them similar.
    if filterCode == "304sstl":
        return "Steel, Stainless (Type 302)"
    return filterCode


def _normalizeTwoFractions(a: float, b: float) -> tuple[float, float]:
    total = a + b
    if total <= 0:
        raise ValueError("Fraction sum must be > 0.")
    return a / total, b / total


def _massFractionsFromVolumeFractions(
    volumeFrac1: float,
    density1: float,
    volumeFrac2: float,
    density2: float,
) -> tuple[float, float]:
    # Mass contribution is proportional to volume * density
    mass1 = volumeFrac1 * density1
    mass2 = volumeFrac2 * density2
    return _normalizeTwoFractions(mass1, mass2)


def _buildEquivalentMaterialProperties(
    material1: str,
    material2: str | None = None,
    volumeFractionMaterial2: float = 0.0,
) -> tuple[list[float], list[str], float, dict]:
    """
    Build equivalent elemental composition + equivalent density for one/two-material mixture.

    Inputs:
      - material1: required
      - material2: optional
      - volumeFractionMaterial2: volume fraction of material2 in [0,1]

    Returns:
      elementWeights: list[float]        # normalized to sum=1
      elementSymbols: list[str]          # same length as elementWeights
      mixtureDensity: float              # g/cm^3 equivalent
      meta: dict                         # debug / UI fields
    """
    if not (0.0 <= volumeFractionMaterial2 <= 1.0):
        raise ValueError("Volume fraction of material 2 must be in [0, 1].")

    useSecondMaterial = (
        material2 is not None
        and material2.strip() != ""
        and volumeFractionMaterial2 > 0.0
    )

    # --- Material 1 properties ---
    w1_elem, s1_elem, rho1 = mpd.getMaterialProperties(material1)
    w1_elem = np.asarray(w1_elem, dtype=float)
    s1_elem = list(s1_elem)
    rho1 = float(rho1)

    if len(w1_elem) != len(s1_elem):
        raise ValueError(f"Material '{material1}' has mismatched symbols/weights.")
    if np.any(w1_elem < 0):
        raise ValueError(f"Material '{material1}' has negative elemental weights.")
    if float(np.sum(w1_elem)) <= 0:
        raise ValueError(f"Material '{material1}' has zero elemental weight sum.")
    w1_elem = w1_elem / float(np.sum(w1_elem))  # safety normalize

    # --- Single material path ---
    if not useSecondMaterial:
        meta = {
            "isTwoMaterial": False,
            "material1": material1,
            "material2": "-",
            "volumeFrac1": 1.0,
            "volumeFrac2": 0.0,
            "massFrac1": 1.0,
            "massFrac2": 0.0,
            "rho1": rho1,
            "rho2": 0.0,
        }
        return w1_elem.tolist(), s1_elem, rho1, meta

    # --- Material 2 properties ---
    w2_elem, s2_elem, rho2 = mpd.getMaterialProperties(material2)
    w2_elem = np.asarray(w2_elem, dtype=float)
    s2_elem = list(s2_elem)
    rho2 = float(rho2)

    if len(w2_elem) != len(s2_elem):
        raise ValueError(f"Material '{material2}' has mismatched symbols/weights.")
    if np.any(w2_elem < 0):
        raise ValueError(f"Material '{material2}' has negative elemental weights.")
    if float(np.sum(w2_elem)) <= 0:
        raise ValueError(f"Material '{material2}' has zero elemental weight sum.")
    w2_elem = w2_elem / float(np.sum(w2_elem))  # safety normalize

    # --- Volume -> mass fractions ---
    v2 = float(volumeFractionMaterial2)
    v1 = 1.0 - v2
    m1 = v1 * rho1
    m2 = v2 * rho2
    msum = m1 + m2
    if msum <= 0:
        raise ValueError("Invalid mixture: mass sum <= 0.")

    massFrac1 = m1 / msum
    massFrac2 = m2 / msum

    # --- Merge elemental composition ---
    # mixture elemental mass fraction:
    #   w_mix[e] = massFrac1 * w1[e] + massFrac2 * w2[e]
    symbolToWeight: dict[str, float] = {}

    for sym, w in zip(s1_elem, w1_elem):
        symbolToWeight[sym] = symbolToWeight.get(sym, 0.0) + massFrac1 * float(w)

    for sym, w in zip(s2_elem, w2_elem):
        symbolToWeight[sym] = symbolToWeight.get(sym, 0.0) + massFrac2 * float(w)

    # stable ordering: by atomic number if possible, else alphabetic fallback
    try:
        elementSymbols = sorted(symbolToWeight.keys(), key=lambda x: xraydb.atomic_number(x))
    except Exception:
        elementSymbols = sorted(symbolToWeight.keys())

    elementWeights = np.asarray([symbolToWeight[s] for s in elementSymbols], dtype=float)

    # final normalization to protect against floating-point drift
    wsum = float(np.sum(elementWeights))
    if wsum <= 0:
        raise ValueError("Mixture elemental weights sum to zero.")
    elementWeights = elementWeights / wsum

    # --- Mixture density ---
    # consistent with your mass-from-volume mixing:
    # density_mix = total_mass / total_volume = v1*rho1 + v2*rho2
    mixtureDensity = v1 * rho1 + v2 * rho2

    meta = {
        "isTwoMaterial": True,
        "material1": material1,
        "material2": material2,
        "volumeFrac1": v1,
        "volumeFrac2": v2,
        "massFrac1": massFrac1,
        "massFrac2": massFrac2,
        "rho1": rho1,
        "rho2": rho2,
    }

    return elementWeights.tolist(), elementSymbols, float(mixtureDensity), meta



def getRuleOfThumbFilterThickness(
    kvp: float,
    filterMaterial: str,
    sampleDiameterMm: float,
    material1: str,
    material2: str | None = None,
    volumeFractionMaterial2: float = 0.0,
) -> float:
    materialWeights, materialSymbols, density, _ = _buildEquivalentMaterialProperties(
        material1=material1,
        material2=material2,
        volumeFractionMaterial2=volumeFractionMaterial2,
    )

    return float(
        fp.getRuleOfThumbFilterThickness(
            kvp=kvp,
            filterMaterial=filterMaterial,
            sampleDiameterMm=sampleDiameterMm,
            materialWeights=materialWeights,
            materialSymbols=materialSymbols,
            density=density,
        )
    )


def runAll(
    kvp: float,
    filterMaterial: str,
    filterThicknessMm: float,
    sampleDiameterMm: float,
    material1: str,
    material2: str | None = None,
    volumeFractionMaterial2: float = 0.0,
    coneAngleDeg: float = 10.0,
    sampleDiameterVx: int = 256,
    tubeThicknessMm: float = 0.0,   # treated as extra filtering
    tubeMaterial: str = "Al",
) -> dict:
    """
    Unified compute for one- or two-material samples.

    Inputs:
      - material1, material2(optional), volumeFractionMaterial2
      - optional tube as extra spectral filtering

    Model:
      - Convert volume fraction -> mass fraction for composite definition
      - Convert composite -> equivalent elemental properties once
      - Use same elemental properties for scatter/BHC/transmission
      - Tube is applied as extra spectral filtering layer
    """
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

    # Equivalent material properties (computed once)
    materialWeights, materialSymbols, density, materialMeta = _buildEquivalentMaterialProperties(material1=material1,
        material2=material2, volumeFractionMaterial2=volumeFractionMaterial2, )

    # 1) Spectrum at detector after selected filter
    energyKeV, spectrumDet = fp.setSpectrum(
        kvp=kvp,
        filterMaterial=filterMaterial,
        filterThicknessMm=filterThicknessMm,
    )

    # 1b) Optional tube as extra filtering layer
    if tubeThicknessMm > 0.0:
        tubeWeights, tubeSymbols, tubeDensity = mpd.getMaterialProperties(tubeMaterial)
        tubeTrans = xs.calcTransmission(
            energyKeV,
            tubeWeights,
            tubeSymbols,
            tubeDensity,
            tubeThicknessMm / 10.0,  # mm -> cm
        )
        spectrumDet = spectrumDet * tubeTrans

    # 2) Scatter % using direct equivalent elemental properties
    scatterPercent = scat.estimateSampleScatterAsPercentOfTransmission(
        energyKeV=energyKeV,
        spectrum=spectrumDet,
        sampleDiameterMm=sampleDiameterMm,
        coneAngleDeg=coneAngleDeg,
        materialWeights=materialWeights,
        materialSymbols=materialSymbols,
        density=density,
    )

    # 3) Beam hardening factor using direct equivalent elemental properties
    bhcFactor, voxelSizeMm, *_ = bhs.simulateBH(
        sampleDiameterMm=sampleDiameterMm,
        sampleDiameterVx=sampleDiameterVx,
        kvp=kvp,
        filterMaterial=filterMaterial,
        filterThicknessMm=filterThicknessMm,
        materialWeights=materialWeights,
        materialSymbols=materialSymbols,
        density=density,
        plotCurve=False,
        plotIdeal=False,
        plotBH=False,
        verbose=False,
    )

    # 4) Transmission and effective attenuation using same properties
    totalTransmission = xs.calcSpecTrans(
        energyKeV,
        spectrumDet,
        materialWeights,
        materialSymbols,
        density,
        sampleDiameterMm,
    )
    muPerCm = -np.log(totalTransmission) / sampleDiameterMm * 10.0

    isTwoMaterial = materialMeta["isTwoMaterial"]

    return {
        "mode": "two-material mixture" if isTwoMaterial else "single material",
        "material 1": material1,
        "material 2": material2 if isTwoMaterial else "-",
        "vol frac material 2": round(float(volumeFractionMaterial2 if isTwoMaterial else 0.0), 4),
        "mass frac material 1": round(float(materialMeta["massFrac1"]), 4),
        "mass frac material 2": round(float(materialMeta["massFrac2"]), 4),
        "density material 1": round(float(materialMeta["rho1"]), 4),
        "density material 2": round(float(materialMeta["rho2"]), 4) if materialMeta["isTwoMaterial"] else 0.0,
        "sample density (equivalent)": round(float(density), 4),
        "filter material": filterMaterial,
        "filter thickness mm": round(float(filterThicknessMm), 4),
        "tube material": tubeMaterial if tubeThicknessMm > 0 else "-",
        "tube thickness mm": round(float(tubeThicknessMm), 4),
        "scatter/transmission in percentage": round(float(scatterPercent), 4),
        "bhc factor": round(float(bhcFactor), 4),
        "transmission percentage": round(float(totalTransmission * 100.0), 4),
        "effective attenuation per cm": round(float(muPerCm), 4),
        # keep if you need debugging:
        # "voxel size mm": round(float(voxelSizeMm), 6),
        # "material weights": materialWeights,
        # "material symbols": materialSymbols,
    }
