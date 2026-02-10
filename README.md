# How to use this software

## To calculate the rule-of-thumb (ROT) thickness of the scan

Fill in the kvp, filter material, material 1, and sample thickness. If there is a second material, can be contrast agent, toggle the option 'enable material 2' and fill in the volume fraction. Then click the 'calculate ROT thickness' button.

You can apply this value directly by clicking the 'apply ROT thickness' button, or manually input another value.

## To calculate other quantities

Cone angle can affect the result for scattering. The default value is given based on the ANU4 geometry, which estimates the scattering from just the sample itself.

If you are using a sample tube, you can toggle the 'enable tube as extra filtering' option and fill in the info.

After you fill in all the values, you can click the 'calculate all' button. The results will be displayed in the output box. You may have to scroll down a bit to see the full results.


# Q&A for the current version

## Q: Does the scatter/transmission mean that if it is 10% that 10% of the 0.5% transmission is probably scatter?

A: Yes, you are right. 

## Q: Is the scatter estimate for the source with an aperture or without? Or is the scatter component only scatter from the sample, or does it include other scatter from the CT cabinet? 

A: The scattering calculated only includes the scattering from the sample itself,  with the part that been reflected by the cabinet and detector table. Assuming the geometry of ANU4. (this can be significantly smaller compared to the actual value, but very close to the topas simulation)

## Q: We should also have a filter setting for some of the "composite equivalent" filters as an option. 

A: Yes, I guess this will not have too much difference on bhc and transmission, but more on scattering, if we include the scattering from the filter.

## Q: When selecting a sample material you are assuming that material solid or has some porosity?

A: Yes, unless you toggle the 'enable material 2' option.



## table of material list (26/08/2025):
Valid materials are: '304sstl', 'acrylic', 'air', 'al', 'al6061', 'caco3', 'carbon fiber', 'carbonate', 'chalcopyrite', 'clastic', 'coal', 'cu', 'dolomite', 'feo', 'glass', 'goethite', 'graphite', 'greenalite', 'h2o', 'haematite', 'hardwood', 'iron ore', 'limestone', 'magnetite', 'marble', 'monzonite', 'nai1.5mol', 'peek', 'pmma', 'pyrex', 'pyrite', 'sandstone', 'si', 'siderite', 'softwood', 'spt0.27mol', 'spt0.63mol', 'spt_saturated', 'teflon', 'ti', 'ti64', 'tin', 'titanium', 'wustite', 'xe', 'xe500psi'
