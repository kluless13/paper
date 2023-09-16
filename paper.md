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

### Discussion
#### Points of Improvement
##### Data Augmentation
- Techniques and benefits.
##### Dataset Improvements
- Recommendations for enhancing dataset quality.
##### Camera Improvements
- Suggestions for hardware enhancements.

### Code and Dataset Availability
- Links and access details.

### Conclusion
- Summary of findings and their implications.

### Future Work
- Potential extensions and areas of exploration.

### Acknowledgements
- Credits and thanks.

---
