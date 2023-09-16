## Invasive Species Detector

### Abstract 
(200 words) 
(Keywords in alphabetical order)

### Introduction
- Background on invasive species and their ecological impact.
- Highlighting COTs (Crown of Thorns starfish) in Australia and Lionfish in the USA as primary examples.
- Objectives:
    - To evaluate different detection models for edge devices.
    - To discuss target types and recommend suitable models.

### Related Work
- Overview of invasive species detection methodologies.
- DeepPlastic: [Link](https://arxiv.org/pdf/2105.01882.pdf)
- Marine object detection techniques.
- CSIRO: COTs detection - [Link](https://arxiv.org/pdf/2111.14311v1.pdf)
- Transfer learning in COTs detection: [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10075425/)
- YOLOv5 for COTs detection: [GitHub Link](https://github.com/SelendisErised/Crown-of-Thorns-Starfish-Detection)
- Lionfish in the US: [Link](https://www.tandfonline.com/doi/full/10.1080/10641262.2012.700655)
- Challenges in Lionfish eradication: [Link](https://www.sciencedirect.com/science/article/pii/S0048969719328554)

### Network Architecture
- Emphasis on real-time detection using YOLOv7 and YOLOv8.
- Comparative analysis of YOLOv7 vs. YOLOv8.
- Detailed description of the differences between the two versions.

### Methodology
#### Dataset Construction
- Description of data sources and collection methods.
#### Enhancements of Custom Datasets
- Techniques used to improve dataset quality.
#### Object Detection
##### Fine-Tuning Parameters
- Parameter adjustments for optimal performance.
##### GPU Hardware
- Hardware specifications and configurations.
##### Training 
- Training methodologies and techniques.
##### Evaluation Metrics
- True Positive (TP) and True Negative (TN) Values
- Precision and Recall
- Mean Average Precision (mAP)
##### Visualizing Results
- Graphs, charts, and other visualization tools -- Table of results + Confusion Matrices

### Results
#### Quantitative Results
- Numerical data and findings.
#### Evaluation Results
##### Object Detection
- Performance of object detection models.
##### Inference Speed 
- Speed of model inference and real-time capabilities.
#### Qualitative Results
- Interpretation and analysis of results.
- Comparison of YOLOv7 and YOLOv8 using mAP score, Precision curve, Recall curve, and PR curve.
- Confusion matrices for each model.

**YOLOv7**

| Dataset/Metric | F1 Score | Precision | Recall | Precision-Recall |
|----------------|----------|-----------|--------|------------------|
| Lionfish       |          |           |        |                  |
| COTs           |          |           |        |                  |

**YOLOv8**

| Dataset/Metric | F1 Score | Precision | Recall | Precision-Recall |
|----------------|----------|-----------|--------|------------------|
| Lionfish       |     0.96     |    1.00   |    0.98    |    0.98     |
| COTs           |          |           |        |                  |

### Discussion
Comments on yolov8 for Lionfish - dataset from Roboflow:

@misc{ lionfish-sserd_dataset,
    title = { lionfish Dataset },
    type = { Open Source Dataset },
    author = { hunter gunter },
    howpublished = { \url{ https://universe.roboflow.com/hunter-gunter/lionfish-sserd } },
    url = { https://universe.roboflow.com/hunter-gunter/lionfish-sserd },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2021 },
    month = { nov },
    note = { visited on 2023-09-16 },
}

75 epochs trained on 8s model, possible overfitting as we only used the model to train Lionfish as a stand alone class, meaning it may confuse other fish for lionfish. However the weights from this model may be used in other datasets to tune models for classification. Our goal was to create a detector for specific species.

Comments on yolov8 for COTS - dataset from Roboflow:

@misc{ cots-hwfzf_dataset,
    title = { COTS Dataset },
    type = { Open Source Dataset },
    author = { COTS starfish },
    howpublished = { \url{ https://universe.roboflow.com/cots-starfish/cots-hwfzf } },
    url = { https://universe.roboflow.com/cots-starfish/cots-hwfzf },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { jul },
    note = { visited on 2023-09-16 },




### Code and Dataset Availability
- Links and access details.

### Conclusion
- Summary of findings and their implications.

### Future Work
- Potential extensions and areas of exploration.

### Acknowledgements
- Credits and thanks.

---
