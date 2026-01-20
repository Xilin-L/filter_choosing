from __future__ import annotations

from core import filterPerformance as fp
from core import scatteringSimulation as scat
from core import beamHardeningSimulation as bhs
from core import xraySimulation as xs
from core import materialPropertiesData as mpd

import numpy as np

FILTER_OPTIONS = ["Al", "Fe", "Cu", "304sstl"]


def toSpekpyFilterMaterial(filterCode: str) -> str:
    # SpekPy has 302; your lab option is 304sstl and you consider them similar.
    if filterCode == "304sstl":
        return "Steel, Stainless (Type 302)"
    return filterCode


def getRuleOfThumbFilterThickness(
    kvp: float,
    filterMaterial: str,
    sampleMaterial: str,
    sampleDiameterMm: float,
) -> float:
    # Independent helper for operators.
    return float(
        fp.getRuleOfThumbFilterThickness(
            kvp=kvp,
            filterMaterial=filterMaterial,
            sampleMaterial=sampleMaterial,
            sampleDiameterMm=sampleDiameterMm,
        )
    )


def runAll(
    machine: str,
    mode: str,
    kvp: float,
    filterMaterial: str,
    filterThicknessMm: float,
    sampleMaterial: str,
    sampleDiameterMm: float,
    # scatter
    coneAngleDeg: float = 10.0,
    # beam hardening sim control
    sampleDiameterVx: int = 256,
    # flux
    sourceCurrentUa: float | None = None,
    powerWatt: float | None = None,
    exposureTimeSec: float = 1.0,
    accumulationNum: int = 1,
) -> dict:
    """
    Runs:
      - detector spectrum (via setSpectrum)
      - scatter percentage (estimateMeasuredScatterAsPercentOfFlux)
      - beam hardening factor (simulateBH)
      - source flux (estimateSourceFlux)

    Returns a dict suitable for UI display/export.
    """

    # 1) Spectrum at detector
    energyKeV, spectrumDet = fp.setSpectrum(
        kvp,
        filterMaterial=filterMaterial,
        filterThicknessMm=filterThicknessMm,
    )

    # 2) Scatter %
    scatterPercent = scat.estimateSampleScatterAsPercentOfTransmission(energyKeV, spectrumDet, sampleMaterial,
                                                                       sampleDiameterMm, coneAngleDeg=coneAngleDeg)

    # 3) Beam hardening factor
    bhcFactor, voxelSizeMm, *_ = bhs.simulateBH(
        sampleDiameterMm=sampleDiameterMm,
        sampleDiameterVx=sampleDiameterVx,
        kvp=kvp,
        filterMaterial=filterMaterial,
        filterThicknessMm=filterThicknessMm,
        materialName=sampleMaterial,
        plotCurve=False,
        plotIdeal=False,
        plotBH=False,
        verbose=False,
    )

    # 4) Source flux (SpekPy expects 302 name for stainless)
    # sourceFlux = xs.estimateSourceFlux(
    #     machine=machine,
    #     mode=mode,
    #     sourceCurrentUa=sourceCurrentUa,
    #     voltageKv=kvp,
    #     powerWatt=powerWatt,
    #     exposureTimeSec=exposureTimeSec,
    #     accumulationNum=accumulationNum,
    #     filterMaterial=toSpekpyFilterMaterial(filterMaterial),
    #     filterThicknessMm=filterThicknessMm,
    # )

    materialWeights, materialSymbols, dens = mpd.getMaterialProperties(sampleMaterial)
    totalTransmission = xs.calcSpecTrans(energyKeV, spectrumDet, materialWeights, materialSymbols, dens, sampleDiameterMm)
    muPerCm = -np.log(totalTransmission) / sampleDiameterMm * 10


    return {
        "scatter/transmission in percentage": float(scatterPercent),
        "bhc factor": float(bhcFactor),
        "transmission percentage": float(totalTransmission*100),
        "effective attenuation per cm": float(muPerCm),

        # "voxelSizeMm": float(voxelSizeMm),
        # "sourceFlux": float(sourceFlux),
        # Optional: include these if you want plotting/export later
        # "energyKeV": energyKeV,
        # "spectrumDet": spectrumDet,
    }
