## Invasive species detector

### Abstract (200 words)(keywords in alphabetical order)

### Introduction
- COTs (Crown of thorns starfish) - pain in Australia
- Lionfish - Pain in the USA

### Related Work
- Work related to invasive species detection
- Other object detection stuff in the marine field
- CSIRO: COTs detection - https://arxiv.org/pdf/2111.14311v1.pdf
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10075425/#:~:text=This%20paper%20proposes%20a%20novel,classifying%20COTS%20using%20transfer%20learning.
(COTS - threatening species in Australia a.k.a Marine Pest - CNN model - can't run in near real time)
- https://github.com/SelendisErised/Crown-of-Thorns-Starfish-Detection (COTS - done with YOLOv5)
- Lionfish: Invasive to the US - https://www.tandfonline.com/doi/full/10.1080/10641262.2012.700655 - direct removal is good.
- Eradication tricky, needs more digging into: https://www.sciencedirect.com/science/article/pii/S0048969719328554

### Network Architecture
- Since it is real time detection we will test this out on five versions of YOLOv7 and YOLOv8, i.e., 10 models (n, s, m, l, x) and weigh their 
pros and cons.

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