Each protein folder contains both the entire analysed edge data ('results') and the computed node results ('ligands').

For the edge data, the files are named as follows:
```final_summary_SOMD_MBAR_alchemlyb_None_eqtrue_statstrue_truncate0_100.csv```
This states the engine, followed by the estimator. `eqtrue` indicates whether or not automatic equilibration detection was used. `statstrue` indicates whether or not subsampling for statistical inefficiency was preformed. `truncate0_100` indicates at which percentages of the full run the data was spliced.

For the ligand data, there are subfolders for each network-wide analysed method described in the main manuscript.
Within the folders, the files are named based on 'MDENGINE_network'.

