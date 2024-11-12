import matplotlib.pyplot as plt
import numpy as np
import xraySimulation as xs
import xrayImagingPerformance as xip
import beamHardeningSimulation as bhs


def bhFactorPlot(material='iron ore', filterthickness=0.5, sampleDiameterMm=10.0):
    kvp_values = range(30, 510, 20)
    bh_factors = []
    transmissions = []
    transmissionTotals = []

    # Function to run the simulation and collect BHC factors and transmission
    def bhFactorCalc(kvp, material, sampleDiameterMm):
        energyKeV, spectrum = xs.generateEmittedSpectrum(kvp)
        spectrumfilt = xip.getFilteredSpectrum(energyKeV, spectrum, 'Fe', filterthickness)
        materialWeights, materialSymbols, dens = xip.getMaterialProperties(material)
        sampleAttPerCm = bhs.getMaterialAttenuation(energyKeV, materialWeights, materialSymbols, dens)
        A, n = xip.estimateBeamHardening(spectrumfilt, sampleAttPerCm, sampleDiameterMm)
        bhc_factor = 1.0 / n

        tSampFiltCm = sampleDiameterMm / 10.0
        sampFiltTrans = xs.calcTransmission(energyKeV, materialWeights, materialSymbols, dens, tSampFiltCm)
        spectrumOut = spectrumfilt * sampFiltTrans
        transmission = np.sum(spectrumOut)/np.sum(spectrumfilt)  # Calculate transmission
        transmissionTotal = np.sum(spectrumOut)/np.sum(spectrum)  # Calculate transmission
        return bhc_factor, transmission, transmissionTotal

    # Collect BHC factors and transmission for each kVp value
    for kvp in kvp_values:
        bhc_factor, transmission, transmissionTotal = bhFactorCalc(kvp, material, sampleDiameterMm)
        bh_factors.append(bhc_factor)
        transmissions.append(transmission)
        transmissionTotals.append(transmissionTotal)
    # Plot the BHC factors and transmission against kVp on the same plot
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('kVp')
    ax1.set_ylabel('BHC Factor', color=color)
    ax1.plot(kvp_values, bh_factors, label='BHC Factor', color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Sample Transmission', color=color)  # we already handled the x-label with ax1
    ax2.plot(kvp_values, transmissions, label='ST', color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)
    ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
    ax3.spines['right'].set_position(('outward', 60))  # move the third y-axis outward
    color = 'tab:green'
    ax3.set_ylabel('Total Transmission', color=color)
    ax3.plot(kvp_values, transmissionTotals, label='TT', color=color, marker='s')
    ax3.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.subplots_adjust(top=0.95)  # Add padding to the top of the plot
    plt.title(f'{sampleDiameterMm} mm {material} with {filterthickness} mm Fe filter')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    for filterthickness in [0]:
        bhFactorPlot('feo', filterthickness, 4)