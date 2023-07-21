# Artifact for "Magnetic Backscatter for In-body Communication and Localization"

This code base provides data and code to reproduce the figures in the paper titled "Magnetic Backscatter for In-body Communication and
Localization" in Mobicom 2023 Madrid, Spain. If you find this repository helpful, kindly cite our paper using the bibtex citation below
'''


'''

# Instructions and Details for the Codebase

1. Create and activate the conda environment from the yaml file in the codebase
```bash
conda env create -f MobicomArtifact.yml
```
2. Download data files from the Box link: https://uofi.box.com/s/jqvbglwv7xc3gi2batane4nv5e0zh4au
3. From the download, set the directory "~/ProjectEspana/" as your working directory
4. The following data folders should be in your working directory "air_newpcb/", "comm20kbps/", "comm500bps/", "commold/", "magneticnonoise/",           "oldpcbporkbr/",   "oldpcbporkbr3/", "comm1kbps/",   "comm2kbps/",   "comm5kbps/", "magneticnoise/",  "new_pork/", "oldpcbairbr/",  "oldpcbporkbr2/",  "old_pcb_pork_lifted_less/"
5. Download and run the following ipynbs in your working directory Communication.ipynb, Data_Mod_Orig.ipynb, Different_Envs.ipynb, Loc_Generalize.ipynb, Microbenchmarks.ipynb contain code to reproduce to the figures in the paper. Simply run all cells in order to generate the figures once the prior steps are finished. At the top of each ipynb lists the figures that the ipynb reproduces in the paper.
6. To reproduce the results in Table 1, follow the instructions located in Microbenchmarks.ipynb.
7. To reproduce the results in Figure 8a, run the HFSS file in the box titled MLoc.aedt.
