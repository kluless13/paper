## Invasive species detector

### Abstract (200 words)(keywords in alphabetical order)

### Introduction
- COTs (Crown of thorns starfish) - pain in Australia
- Lionfish - Pain in the USA
- The paper underscores the ecological challenges posed by invasive species, using COTs in Australia and Lionfish in the USA as primary examples.
- Objectives:
    - Best model for each of the two case studies

### Related Work
- Work related to invasive species detection
- DeepPlastic: https://arxiv.org/pdf/2105.01882.pdf
- Other object detection stuff in the marine field
- CSIRO: COTs detection - https://arxiv.org/pdf/2111.14311v1.pdf
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10075425/#:~:text=This%20paper%20proposes%20a%20novel,classifying%20COTS%20using%20transfer%20learning.
(COTS - threatening species in Australia a.k.a Marine Pest - CNN model - can't run in near real time)
- https://github.com/SelendisErised/Crown-of-Thorns-Starfish-Detection (COTS - done with YOLOv5)
- Lionfish: Invasive to the US - https://www.tandfonline.com/doi/full/10.1080/10641262.2012.700655 - direct removal is good.
- Eradication tricky, needs more digging into: https://www.sciencedirect.com/science/article/pii/S0048969719328554
- The paper reviews previous work done in the domain of invasive species detection, with an emphasis on object detection methods for marine environments.
- Specific references include studies on COTs detection, techniques that utilize Convolutional Neural Networks (CNNs), and the adoption of YOLOv5 for COTs detection. 
- The challenge seems to be the lack of real-time processing with some existing models.
- There's also a mention of the invasive nature of Lionfish in the US, the challenge in eradicating them, and potential mitigation strategies.

### Network Architecture
- Since it is real time detection we will test this out on five versions of YOLOv7 and YOLOv8, i.e., 10 models (w6, e6, d6, e6e) & (n, s, m, l, x) and weigh their pros and cons.
- The research seeks to explore the real-time detection capabilities of five versions each of YOLOv7 and YOLOv8. 
- The intention here is a comparative analysis of the strengths and weaknesses of each version.

### Methodology
#### Dataset Construction
#### Enhancements of Custom Datasets
#### Object Detection
##### Fine Tuning Parameters
##### GPU Hardware
##### Training 
##### Evaluation Metrics
TP and TN Values
Precision and Recall
Mean Average Precision (mAP)
##### Visualizing Results

### Results
#### Quantitative Results
#### Evaluation Results
##### Object Detection
##### Inference Speed 
#### Qualitative Results

### Discussion
#### Points of Improvement
##### Data Augmentation
##### Dataset Improvements
##### Camera Improvements

### Code and Dataset Availability

### Conclusion

### Future Work

### Acknowledgements 