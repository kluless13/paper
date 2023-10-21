## Invasive Species Detector

### Abstract 

(200 words) 
(Keywords in alphabetical order)

### Introduction

The emergence of machine learning (ML) and deep learning (DL) technologies has sparked a new era of scientific investigation in various fields, including marine ecology, which has gained particular attention *[1,2]*.The effectiveness of computational tools in efficiently and precisely analyzing large datasets has empowered marine ecologists to develop novel tools, thereby enhancing our comprehension and control of marine ecosystems. The issue of invasive species poses a significant threat to marine ecosystems, adding to the numerous challenges they already face. The introduction of non-native species into new habitats can have significant impacts on the ecological equilibrium, leading to negative consequences for the environment and economy.

In marine ecosystems, the presence of invasive species has been observed to disrupt ecological relationships, resulting in negative consequences. For example, there has been a correlation between the occurrence of outbreaks of Crown of Thorns starfish (COTs) and a notable decrease in coral abundance throughout the Great Barrier Reef over the course of the last forty years. When starfish populations reach a state of outbreak, they exhibit a consumption rate of coral tissues that exceeds the rate of coral growth. This poses a substantial risk to the long-term health and resilience of the reef, particularly in light of climate change *[3,4,5]*. In the same manner, it's also significant that the Lionfish has rapidly infiltrated the Atlantic coastal waters, the Caribbean Sea, and the Gulf of Mexico within the United States. This phenomenon has raised significant apprehension due to their predatory behavior towards indigenous fish species and their absence of natural predators, thereby intensifying their invasive capabilities *(6,7)*.

The cybernetic perspective, which conceptualizes ecosystems as systems that process information, presents a new paradigm for understanding and tackling the challenges posed by invasive species. The utilization of machine learning (ML) and deep learning (DL) algorithms holds the potential to decipher the complex information networks present in marine ecosystems, thus enabling the creation of tools for real-time monitoring and management. These tools play a crucial role in the timely identification and management of invasive species risks, thereby bolstering the ability of marine ecosystems to withstand and recover from such invasions *(8)*.

The study aims to achieve two primary objectives:

- The first aim is to highlight the differences between the YOLOv7 and YOLOv8 model inferences. Edge devices, due to their inherent ability to perform data processing at the location of data collection, present a highly promising opportunity for the real-time monitoring of marine ecosystems.

- The second objective of this study is to examine the different types of targets found in marine ecosystems and propose appropriate models for object detection in each case. The overall goal of this study is to provide marine ecologists with a comprehensive toolkit that can be customized to suit specific monitoring and management goals, through a thorough understanding of the strengths and weaknesses inherent in various models.

This study aims to bridge the gap between marine science and machine learning by conducting a comparative analysis of YOLOv7 and YOLOv8 in the detection and identification of invasive species, specifically COTs and Lionfish. Furthermore, this study provides a foundation for future investigations focused on exploring the cybernetic characteristics of marine ecosystems, thereby enhancing our comprehensive comprehension of the interconnectedness and reciprocal relationships intrinsic to these intricate ecological networks *(9)*. 


1. https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14061#:~:text=Overview%20of%20common%20supervised%20machine,1
2. https://academic.oup.com/icesjms/article/79/2/319/6507793
3. https://www2.gbrmpa.gov.au/our-work/programs-and-projects/crown-thorns-starfish-management
4. https://www.barrierreef.org/the-reef/threats/Crown-of-thorns-starfish#:~:text=An%20adult%20crown,humphead%20maori%20wrasse%2C%20yellow
5. https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.4580#:~:text=Crown,multiple%20stressors%2C%20especially%20anthropogenic%20warming
6. https://www.wuwf.org/local-news/2023-05-22/the-threats-of-lionfish-on-the-gulf-of-mexico
7. https://www.usgs.gov/media/images/1985-2018-lionfish-invasion#:~:text=Lionfish%20invasion%20%281985,case%20of%20a%20non
8. https://academic.oup.com/icesjms/article/79/2/319/6507793
9. https://www.frontiersin.org/articles/10.3389/fmars.2022.920994/full)

---

### Related Work

**DeepPlastic:

This paper discusses a deep learning-based approach for detecting marine litter from aerial images. Although not directly related to invasive species, this work showcases the broader application of machine learning in marine ecology *[1]*.

**CSIRO: COTs detection:

The provided link leads to a paper discussing the use of machine learning for detecting Crown-of-Thorns Starfish (COTs) *[2]*. Additionally, a project called "COTSBot" was developed by Matthew Dunbabin and Feras Dayoub to combat the outbreak of COTs, acting as a "Starfish Terminator" robot *[3]*.

**Lionfish in the US:

The USGS mentions efforts in early detection and rapid response to manage invasive marine fishes in Florida's coastal waters, including Lionfish *[4]*. Additionally, the economic impact of Lionfish invasion is discussed, with estimated losses of US$24 million per year due to the expansion of Lionfish into non-native ecosystems *[5]*.

**Jellyfish Monitoring:

A paper titled "Real-time jellyfish classification and detection algorithm based on improved YOLOv4-tiny and improved underwater image enhancement algorithm" discusses a real-time jellyfish classification and detection algorithm, showcasing advancements in jellyfish monitoring *[6]*. Another paper, "Jellytoring: Real-Time Jellyfish Monitoring", discusses a rise in jellyfish populations and the importance of monitoring these changes for environmental assessment and management *[7]*.

1. https://arxiv.org/pdf/2105.01882.pdf
2. https://arxiv.org/pdf/2111.14311v1.pdf
3. https://www.asme.org/topics-resources/content/underwater-drone-hunts-coraleating-crownofthorns#:~:text=Dunbabin%20and%20Dayoub%20built%20it,creators%20opted%20for%20COTSBot
4. https://www.usgs.gov/centers/wetland-and-aquatic-research-center/science/science-topics/lionfish#:~:text=Advanced%20options%20February%2023%2C%202021,By
5. https://www.sciencedirect.com/science/article/abs/pii/S0301479723007429#:~:text=Early%20detection%20also%20represents%20saving,invasive%20species%20may%20cause
6. https://www.nature.com/articles/s41598-023-39851-7#:~:text=Published%3A%2010%20August%202023%20Real,algorithm%20Meijing%20Gao%2C%20Shiyu%20Li
7. https://www.mdpi.com/1424-8220/20/6/1708/htm#:~:text=During%20the%20past%20decades%2C%20the,a%20global%20scale%2C%20negatively

---

### Network Architecture

- Emphasis on real-time detection using YOLOv7 and YOLOv8.
- Comparative analysis of YOLOv7 vs. YOLOv8.
- Detailed description of the differences between the two versions.


**Emphasis on real-time detection using YOLOv7 and YOLOv8**:

Both YOLOv7 and YOLOv8 are object detection models utilized in computer vision projects, offering real-time detection capabilities *[1]*. Their architecture, grounded in the You Only Look Once (YOLO) framework and Convolutional Neural Networks (CNN), allows for rapid processing and analysis of images, facilitating real-time identification of objects within marine environments *[2,3,4]*.

**Comparative analysis of YOLOv7 vs. YOLOv8**:

YOLOv8 advances over YOLOv7 by enhancing the speed of detection, thereby achieving a faster Frames Per Second (FPS) rate which is pivotal for real-time object detection applications *[2]*. This speed enhancement does not compromise the model's accuracy, showcasing YOLOv8's superior efficiency in real-time object detection scenarios.

**Detailed description of the differences between the two versions**:

YOLOv8 was released by Ultralytics, the developers of YOLOv5, as a state-of-the-art object detection and image segmentation model following YOLOv7 *[1,3]*. The comparison between YOLOv7 and YOLOv8 mainly revolves around the improvements in speed and real-time object detection efficiency, with YOLOv8 standing out due to its faster FPS rate *[2,4]*.

1. https://roboflow.com/compare/yolov8-vs-yolov7#:~:text=,Create%20a%20Confusion%20Matrix%20YOLOv7
2. https://www.augmentedstartups.com/blog/unlock-the-full-potential-of-object-detection-with-yolov8-faster-and-more-accurate-than-yolov7-2#:~:text=,it%20is%20on%20the%20CPU
3. https://blog.roboflow.com/yolov7-breakdown/#:~:text=In%20this%20post%2C%20we%20break,art%20in%20object%20detection
4. https://docs.ultralytics.com/models/yolov7/

---

### Methodology

methods > workflow design - infographic of pipeline > dataset prep > network arch > training > validation > test > model evaluation

#### Dataset Construction

- **Description of data sources and collection methods**:
    - The datasets utilized in this study were meticulously curated from Roboflow, an open-source platform dedicated to amassing and distributing machine learning datasets (citation for the site). Two distinct datasets, one for Lionfish and another for Crown of Thorns starfish (COTs), were assembled to facilitate the comparative analysis of YOLOv7 and YOLOv8's performance in species identification.
    - **Lionfish Dataset**:
        - The Lionfish dataset comprises a total of 955 images, distributed across training, validation, and test sets. Specifically, it encompasses 786 images for the training set, 112 images for the validation set, and 57 images for the test set. The delineation of these datasets is vital for ensuring a robust evaluation of the models' performance across different stages of machine learning, from training through to testing.
    - **COTs Dataset**:
        - Similarly, the COTs dataset encapsulates a total of 3361 images, with 3082 images allocated for the training set, 227 images for the validation set, and 52 images for the test set. This segmentation allows for a comprehensive assessment of the models' proficiency in identifying and counting COTs across varying marine habitats.

#### Enhancements of Custom Datasets

- **Techniques used to improve dataset quality**:
    - Prior to model training, the datasets underwent several preprocessing and augmentation steps aimed at enhancing their quality and increasing the robustness of the models to variations in input data.
    - **Lionfish Dataset**:
        - The images were initially preprocessed by auto-orienting them to ensure a consistent orientation across the dataset. Subsequently, resizing was performed on the images by stretching them to a resolution of 1028x1028 pixels, which is conducive for retaining essential features necessary for accurate species identification. Additionally, augmentations were applied to diversify the dataset and improve model generalization. Specifically, brightness augmentations were executed, varying the brightness levels between -30% and +30% to simulate varying lighting conditions encountered in marine environments.
    - **COTs Dataset**:
        - Analogous to the Lionfish dataset, the images in the COTs dataset were auto-oriented. Resizing was conducted by stretching the images to a resolution of 416x416 pixels, which balances the trade-off between resolution and computational efficiency. The augmentations entailed Shear transformations of ±15° both horizontally and vertically to simulate possible variations in perspective, and Cutout augmentations with 5 boxes of 12% size each were applied to promote model robustness against occlusions and varying field-of-view scenarios.

#### Model Training and Evaluation

- **Training Procedure**:
    - Following the dataset construction and enhancement, both YOLOv7 and YOLOv8s models were trained on the Lionfish and COTs datasets. The training procedure was conducted with differing epochs for each model to optimize their learning and generalization capabilities. Specifically:
        - The YOLOv8s models were trained for 75 epochs, which allowed for a sufficient number of iterations for the models to learn and generalize well across the datasets.
        - The YOLOv7 models, on the other hand, were trained for 55 epochs, aligning with the model’s architectural and computational considerations.
    - The training process encompassed various stages including the initial training on the training set, validation on the validation set, and fine-tuning to enhance the models' performance and accuracy.

- **Model Testing**:
    - Post training, the models were rigorously tested on the designated test sets from the Lionfish and COTs datasets. The testing phase aimed at evaluating the models’ accuracy, speed, and real-time detection capabilities in a controlled setting, providing a benchmark for their performance.

- **Real-world Video Analysis**:
    - Subsequent to the testing phase, the trained models were deployed on real-world video data of Lionfish and COTs. This step aimed at assessing the models' effectiveness and reliability in real-world, dynamic marine environments. The video analysis also provided insights into the models’ real-time detection and tracking capabilities, thereby demonstrating their potential utility in ongoing and future marine ecology studies.

---

### Results

The evaluation of the models' performance is a crucial step towards understanding their effectiveness and applicability in real-world marine ecological scenarios. This section elucidates the quantitative and qualitative evaluation results obtained post the training and testing phases.

#### Quantitative Results

- **Numerical data and findings**:
    - The quantitative evaluation primarily focused on understanding the models' performance in terms of their accuracy, speed, and real-time detection capabilities. The numerical findings provide a robust basis for comparing the models and understanding their strengths and weaknesses in different scenarios.

#### Evaluation Metrics

A comprehensive evaluation encompassing various metrics was conducted to ensure a holistic understanding of the models' performance on the Lionfish and COTs datasets. The metrics evaluated include F1 Score, Precision, Recall, and Precision-Recall.

#### YOLOv7 Evaluation

The table below elucidates the performance of the YOLOv7 model on both datasets:

| Dataset/Metric | F1 Score | Precision | Recall | Precision-Recall |
|:--------------:|:--------:|:---------:|:------:|:----------------:|
| Lionfish       |   0.95   |   1.00    |  1.00  |       0.97       |
| COTs           |   0.82   |   1.00    |  1.00  |       0.88       |

The YOLOv7 model exhibited stellar performance on the Lionfish dataset, securing an F1 Score of 0.95. However, a discernible decline in performance was observed on the COTs dataset, where it secured an F1 Score of 0.82. The consistency in Precision and Recall values across the datasets highlights the model's robustness, albeit with room for improvement in handling COTs detection.

#### YOLOv8 Evaluation

The subsequent table delineates the performance metrics of the YOLOv8 model:

| Dataset/Metric | F1 Score | Precision | Recall | Precision-Recall |
|:--------------:|:--------:|:---------:|:------:|:----------------:|
| Lionfish       |   0.96   |   1.00    |  0.98  |      0.98        |
| COTs           |   0.92   |   1.00    |  0.97  |      0.95        |

In contrast, the YOLOv8 model showcased a more consistent performance across both datasets, with a marginal difference in F1 Scores between Lionfish and COTs. The high Precision and Recall values depict the model's superior capability in accurately identifying and classifying both Lionfish and COTs, making it a viable candidate for real-world marine species identification tasks.

YOLO-NAS done for both - confusion matrices in appendix.

#### Confusion martices

Confusion matrices are pivotal in understanding the performance of classification models as they provide a clear visualization of the models' predictions in comparison to the true labels. They encapsulate the true positive, true negative, false positive, and false negative values, offering a comprehensive view of the models’ predictive accuracy and misclassifications. In this section, we delve into the true positive prediction confidence exhibited by YOLOv7 and YOLOv8 on both the Crown Of Thorns starfish (COTs) and Lionfish datasets.

- **YOLOv7 Evaluation**:
    - **COTs Dataset**:
        - The YOLOv7 model exhibited a true positive prediction confidence of 89% on the COTs dataset. This implies a high degree of accuracy in correctly identifying and classifying the Crown Of Thorns starfish, though there is still an 11% margin where the model could potentially misclassify or fail to detect the starfish.
    - **Lionfish Dataset**:
        - On the Lionfish dataset, YOLOv7 demonstrated a slightly higher true positive prediction confidence of 94%. This denotes a commendable level of accuracy in distinguishing Lionfish from other marine species and objects within the dataset.

- **YOLOv8 Evaluation**:
    - **COTs Dataset**:
        - With the YOLOv8 model, a true positive prediction confidence of 95% was achieved on the COTs dataset. This marginal improvement over YOLOv7 highlights YOLOv8's refined capability in accurately identifying the Crown Of Thorns starfish.
    - **Lionfish Dataset**:
        - Similarly, on the Lionfish dataset, YOLOv8 exhibited a true positive prediction confidence of 94%, matching the performance of YOLOv7. This consistency in true positive prediction confidence across the two models for Lionfish detection underscores a robust level of accuracy and reliability.

---

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
