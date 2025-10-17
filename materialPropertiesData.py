
import numpy as np

def getMaterialProperties(material):
    """
    Retrieves material properties including elemental weight fractions, element symbols, and density.
    Supports both single materials and composite materials with strict, case-sensitive material names.

    Parameters:
    - material (str or list):
        - If str: Name of the material (e.g., "sandstone").
        - If list:
            - First element: List of material names (exact case).
            - Second element: List of corresponding percentages (should sum to 1).

    Returns:
    - materialWeights (list): Combined weight fractions of elements.
    - materialSymbols (list): Corresponding element symbols.
    - dens (float): Combined density of the material.

    Raises:
    - Exception: If an unknown material is provided, input format is incorrect, or percentages are invalid.
    """

    if isinstance(material, str):
        # Single material
        material = material.lower()
        if material in materialProperties:
            props = materialProperties[material]
            return props["weights"], props["symbols"], props["density"]
        elif material in materialNameMapping:
            canonicalName = materialNameMapping[material]
            props = materialProperties[canonicalName]
            return props["weights"], props["symbols"], props["density"]
        else:
            validMaterials = list(materialProperties.keys()) + list(materialNameMapping.keys())
            raise Exception(f"Unknown sample material: '{material}'. "
                            f"Valid materials are: {', '.join(validMaterials)}.")

    elif isinstance(material, list):
        if len(material) != 2:
            raise Exception("Composite material input must be a list of two lists: [materials, percentages].")

        materialsList, percentagesList = material

        if not (isinstance(materialsList, list) and isinstance(percentagesList, list)):
            raise Exception("Composite material input must be a list of two lists: [materials, percentages].")

        if len(materialsList) != len(percentagesList):
            raise Exception("Materials list and percentages list must have the same length.")

        totalPercentage = sum(percentagesList)
        if not np.isclose(totalPercentage, 1.0):
            raise Exception(f"Percentages must sum to 1.0, but sum to {totalPercentage}.")

        # Initialize dictionaries to accumulate element weights
        combinedElements = {}
        combinedDensity = 0.0

        for mat, perc in zip(materialsList, percentagesList):
            mat = mat.lower()
            if mat in materialProperties:
                props = materialProperties[mat]
            elif mat in materialNameMapping:
                canonicalName = materialNameMapping[mat]
                props = materialProperties[canonicalName]
            else:
                validMaterials = list(materialProperties.keys()) + list(materialNameMapping.keys())
                raise Exception(f"Unknown sample material: '{mat}'. "
                                f"Valid materials are: {', '.join(validMaterials)}.")

            matWeights, matSymbols, matDens = props["weights"], props["symbols"], props["density"]

            # Accumulate density as weighted average
            combinedDensity += perc * matDens

            # Accumulate element weights
            for w, sym in zip(matWeights, matSymbols):
                if sym in combinedElements:
                    combinedElements[sym] += perc * w
                else:
                    combinedElements[sym] = perc * w

        # Normalize the combined element weights to sum to 1
        totalElementWeight = sum(combinedElements.values())
        if totalElementWeight == 0:
            raise Exception("Total element weight is zero. Check material weights and percentages.")
        for sym in combinedElements:
            combinedElements[sym] /= totalElementWeight

        # Sort elements alphabetically for consistency
        sortedElements = sorted(combinedElements.items())
        materialSymbols = [sym for sym, _ in sortedElements]
        materialWeights = [w for _, w in sortedElements]

        return materialWeights, materialSymbols, combinedDensity

    else:
        raise Exception("Input material must be either a string or a list of two lists [materials, percentages].")

materialNameMapping = {
    "clastic": "sandstone",
    "glass": "sandstone",
    "carbonate": "limestone",
    "marble": "limestone",
    "caco3": "limestone",
    "titanium": "ti",
    "wustite": "feo",
    "acrylic": "pmma",
}

materialProperties = {
    "sandstone": {
        "weights": [0.532565, 0.467435],
        "symbols": ["O", "Si"],
        "density": 2.65
    },
    "limestone": {
        "weights": [0.120005, 0.479564, 0.400431],
        "symbols": ["C", "O", "Ca"],
        "density": 2.71
    },
    "haematite": {
        "weights": [0.300567, 0.699433],
        "symbols": ["O", "Fe"],
        "density": 5.3
    },
    "goethite": {
        "weights": [0.011344, 0.360129, 0.628527],
        "symbols": ["H", "O", "Fe"],
        "density": 3.8
    },
    "iron ore": {
        "weights": [0.00473705, 0.32543904, 0.6698239 ],
        "symbols": ["H", "O", "Fe"],
        "density": 4.55
    },
    "feo": {
        "weights": [0.223, 0.777],
        "symbols": ["O", "Fe"],
        "density": 5.745
    },
    "peek": {
        "weights": [0.041948, 0.791569, 0.166483],
        "symbols": ["H", "C", "O"],
        "density": 1.32
    },
    "al": {
        "weights": [1.0],
        "symbols": ["Al"],
        "density": 2.70
    },
    "xe": {
        "weights": [1.0],
        "symbols": ["Xe"],
        "density": 0.00589
    },
    "ti64": {
        "weights": [0.06, 0.90, 0.04],
        "symbols": ["Al", "Ti", "V"],
        "density": 4.43
    },
    "hardwood": {# Jarrah wood, not dried
        "weights": [0.06, 0.52, 0.42],
        "symbols": ["H", "C", "O"],
        "density": 0.85
    },
    "softwood": {# Pine, not dried
        "weights": [0.06, 0.52, 0.42],
        "symbols": ["H", "C", "O"],
        "density": 0.50
    },
    "ti": {
        "weights": [1.0],
        "symbols": ["Ti"],
        "density": 4.51
    },
    "pmma": {
        "weights": [0.080541, 0.599846, 0.319613],
        "symbols": ["H", "C", "O"],
        "density": 1.18
    },
    "greenalite": {
        "weights": [0.0094, 0.1744, 0.3748, 0.4414],
        "symbols": ["H", "Si", "O", "Fe"],
        "density": 3.08
    },
    "dolomite": {
        "weights": [0.5206, 0.1303, 0.2173, 0.1318],
        "symbols": ["C", "O", "Ca", "Mg"],
        "density": 2.84
    },
    "pyrex": {
        "weights": [0.540, 0.377, 0.04, 0.028, 0.011, 0.003],
        "symbols": ["O", "Si", "B", "Na", "Al", "K"],
        "density": 2.23
    },
    "teflon": {
        "weights": [0.760, 0.240],
        "symbols": ["F", "C"],
        "density": 2.20
    },
    "siderite": {
        "weights": [0.482, 0.1037, 0.4143],
        "symbols": ["Fe", "C", "O"],
        "density": 3.96
    },
    "pyrite": {
        "weights": [0.4655, 0.5345],
        "symbols": ["Fe", "S"],
        "density": 5.01
    },
    "chalcopyrite": {
        "weights": [0.3043, 0.3463, 0.3494],
        "symbols": ["Fe", "Cu", "S"],
        "density": 4.19
    },
    "magnetite": {
        "weights": [0.7236, 0.2764],
        "symbols": ["Fe", "O"],
        "density": 5.15
    },
    "xe500psi": {
        "weights": [1.0],
        "symbols": ["Xe"],
        "density": 0.24
    },
    "air": {
        "weights": [0.755, 0.232, 0.013],
        "symbols": ["N", "O", "Ar"],
        "density": 0.0012
    },
    "nai1.5mol": {
        "weights": [0.153, 0.847],
        "symbols": ["Na", "I"],
        "density": 1.16
    },
    "spt": {
        "weights": [0.046, 0.214, 0.740],
        "symbols": ["Na", "O", "W"],
        "density": 3.1
    },
    "spt0.63mol": {
        "weights": [0.046, 0.214, 0.740],
        "symbols": ["Na", "O", "W"],
        "density": 2.65
    },
    "spt0.27mol": {
        "weights": [0.022, 0.569, 0.35, 0.059],
        "symbols": ["Na", "O", "W", "H"],
        "density": 1.7
    },
    "badge": {
        "weights": [0.741, 0.188, 0.071],
        "symbols": ["C", "O", "H"],
        "density": 1.16
    },
    "ha50": {
        "weights": [0.107, 0.868, 0.017, 0.008],
        "symbols": ["H", "O", "Ca", "P"],
        "density": 1.19
    },
    "ha100": {
        "weights": [0.103, 0.849, 0.033, 0.015],
        "symbols": ["H", "O", "Ca", "P"],
        "density": 1.22
    },
    "ha200": {
        "weights": [0.095, 0.814, 0.062, 0.029],
        "symbols": ["H", "O", "Ca", "P"],
        "density": 1.28
    },
    "ha400": {
        "weights": [0.081, 0.756, 0.111, 0.052],
        "symbols": ["H", "O", "Ca", "P"],
        "density": 1.42
    },
    "ha800": {
        "weights": [0.059, 0.659, 0.193, 0.089],
        "symbols": ["H", "O", "Ca", "P"],
        "density": 1.66
    },
    "ha1200": {
        "weights": [0.043, 0.590, 0.251, 0.116],
        "symbols": ["H", "O", "Ca", "P"],
        "density": 1.91
    },
    "304sstl": {
        "weights": [0.001, 0.001, 0.002, 0.18, 0.08, 0.74],
        "symbols": ["C", "P", "S", "Cr", "Ni", "Fe"],
        "density": 7.93
    },
    "h2o": {
        "weights": [0.111894, 0.888106],
        "symbols": ["H", "O"],
        "density": 1.0
    },
    "tin": {
        "weights": [1.0],
        "symbols": ["Sn"],
        "density": 7.3
    },
    "si": {
        "weights": [1.0],
        "symbols": ["Si"],
        "density": 2.33
    },
}