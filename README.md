# precursor-alignment

This provides the supporting code for **Ionization Product Clustering to Improve Peak Alignment in LC-MS-based Metabolomics**.

The basic idea is to cluster related peaks in LC-MS data into ionization product (IP) peaks, and to perform alignment based on the matching of these IP peaks. Our experiments show that an improved alignment performance can be obtained from this method.

Demonstration on how to use to codes are available from [the following demo Jupyter notebook](demo/demo.ipynb). Additionally, the codes can also be ran in stand-alone python script, as shown in the following [demo.py](demo/demo.py) script.

Data used for the alignment experiments (and their associated ground truths) in the paper can be found in the **input** folder.