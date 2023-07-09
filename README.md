# Artifact for "Magnetic Backscatter for In-body Communication and Localization"

This code base provides data and code to reproduce the figures in the paper titled "Magnetic Backscatter for In-body Communication and
Localization" in Mobicom 2023 Madrid, Spain. If you find this repository helpful, kindly cite our paper using the bibtex citation below
'''


'''

# Instructions and Details for the Codebase

1. Create and activate the conda environment from the yaml file in the codebase
'''bash
conda env create -f MobicomArtifact.yml
'''

2. Create a directory in your home folder titled "~/ProjectEspana/"
3. Download the datafolders into the ProjectEspana/ folder: these datafolders should be "air_newpcb/", "comm20kbps/", "comm500bps/", "commold/", "magneticnonoise/",           "oldpcbporkbr/",   "oldpcbporkbr3/", "comm1kbps/",   "comm2kbps/",   "comm5kbps/", "magneticnoise/",  "new_pork/", "oldpcbairbr/",  "oldpcbporkbr2/",  "old_pcb_pork_lifted_less/"
4. The following ipynbs Communication.ipynb, Data_Mod_Orig.ipynb, Different_Envs.ipynb, Loc_Generalize.ipynb, Microbenchmarks.ipynb contain code to reproduce to the figures in the paper. Simply run all cells in order to generate the figures once the prior steps are finished.
5. To reproduce the results in Table 1, follow the instructions located in Microbenchmarks.ipynb.
