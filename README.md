# Interpretation of British Geological Survey Borehole records
This repository contains all of the project code and outputs. It does *not* contain any of the BGS borehole records belonging to the dataset used in this research.

## Folders
```
MSc-project
│   data - all saved files
│   └───lines-v2.txt                - provided ground truth labels
│   └───pre-classifier-predictions  - saved pre-classifier-predictions
│   └───HTR-predictions             - saved HTR predictions
│
└───src  - all project code
│   └───downloader.py               - BGS borehole records download
│   └───TextLineSegment             - text-lines segmentation
│   └───Pre-classifier              - segmentations classification
│   └───TextWordHTR                 - handwritten text word recognition
│   └───TextLineSegment             - handwritten text line recognition
│   └───Evaluation                  - accuracy calculation
```
