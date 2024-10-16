## This directory contains the scripts and data for reproduce the results in 'uncertainty in chemical reaction dynamics' section of our manuscript 'uncertainty qualification for deep learning-based elementary reaction property prediction'
`MeAcet_rxn_info.yaml` records the QM calculated free energies.  
`MeAcet_ml_est.yaml` records the ML model predicted free energies.  
`pyrenex` is a package for parsing elementary reactions to kinetic models, simulating the ideal rectors, and analyzing reaction network.
`pyrenetx` is under developing and will have its own publication in the future. To use `pyrenex`, you should install `pyomo=6.62` and `networkx=3.1`, and extract the pyrenetx in the current file.
The codes reproducing the results in 'uncertainty in chemical reaction dynamics' were provided in `reaction_network_exploration.ipynb`  
`rmg_input.py` is the the script for generating the reaction network mentioned in 'uncertainty in chemical reaction dynamics' section.
