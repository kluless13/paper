## Invasive Species Detector

### Abstract 

(200 words) 
(Keywords in alphabetical order)

### Introduction

- Background on invasive species and their ecological impact.
- COTS are a pest, effect of marine pests on ecosystems.
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
The datasets were collected from Roboflow (citation for the site). The lionfish dataset had 786 images for the training set, 112 images for the validation set and 57 images for the test set. 
For the COTS dataset, the training set had 3082 images, 227 images in the validation set and 52 images for the test set.

#### Enhancements of Custom Datasets

- Techniques used to improve dataset quality.
For the Lionfish dataset -- the images were preprocessed by auto-orienting the images and resizing was done to the images by stretching to 1028x1028. The augmentations applied were of brightness (between -30% and +30%). 
For the COTS dataset -- the images were auto-oriented and resizing was done to the images by stretching to 416x416. The augmentations applied were as follows: Shear: ±15° Horizontal, ±15° Vertical; Cutout: 5 boxes with 12% size each.


### Results

#### Quantitative Results
- Numerical data and findings.

#### Evaluation Results

In this section, we present the evaluation metrics for our models trained on the Lionfish and COTs datasets. The metrics include F1 Score, Precision, Recall, and Precision-Recall.

#### YOLOv7 Evaluation

The table below showcases the performance of the YOLOv7 model on both datasets:

| Dataset/Metric | F1 Score | Precision | Recall | Precision-Recall |
|:--------------:|:--------:|:---------:|:------:|:----------------:|
| Lionfish       |   0.95   |   1.00    |  1.00  |       0.97       |
| COTs           |   0.82   |   1.00    |  1.00  |       0.88       |

From the results, it's evident that the YOLOv7 model performs exceptionally well on the Lionfish dataset, achieving an F1 Score of 0.95. However, there's a noticeable drop in performance on the COTs dataset, with an F1 Score of 0.82.

#### YOLOv8 Evaluation

The subsequent table represents the performance metrics of the YOLOv8 model:

| Dataset/Metric | F1 Score | Precision | Recall | Precision-Recall |
|:--------------:|:--------:|:---------:|:------:|:----------------:|
| Lionfish       |   0.96   |   1.00    |  0.98  |      0.98        |
| COTs           |   0.92   |   1.00    |  0.97  |      0.95        |

Comparatively, the YOLOv8 model exhibits a consistent performance across both datasets, with only a slight difference in F1 Scores between Lionfish and COTs.

YOLO-NAS done for both - confusion matrices in appendix.
---

##### Object Detection
- Performance of object detection models.

##### Inference Speed 
- Speed of model inference and real-time capabilities.

#### Qualitative Results
- Interpretation and analysis of results.
- Comparison of YOLOv7 and YOLOv8 using F1 score, Precision curve, Recall curve, and PR curve.
- Confusion matrices for each model.

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

@misc{ google-images-ztm4n_dataset,
    title = { Google Images Dataset },
    type = { Open Source Dataset },
    author = { COTS },
    howpublished = { \url{ https://universe.roboflow.com/cots/google-images-ztm4n } },
    url = { https://universe.roboflow.com/cots/google-images-ztm4n },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { may },
    note = { visited on 2023-09-16 },
}

75 epochs trained on the 8s model


### Code and Dataset Availability
- Links and access details.

### Conclusion
- Summary of findings and their implications.

### Future Work
- Potential extensions and areas of exploration.

### Acknowledgements
- Credits and thanks.

---
