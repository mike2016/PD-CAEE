# PD-CAEE (Phase Diagram-based Convolutional Auto-Encoder with Ellipse constraint)
Source code for the paper "Hybrid model of Convolutional Auto-Encoder and Ellipse Characteristic for Unsupervised High Impedance Fault Detection"

### IEEE 34-nodde test feeder simulation data is available here:
https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/KRYCYY

YANG, Junjie; DELINCHANT Benoit, 2024, "IEEE 34 Nodes Test Feeder Simulation Data for High Impedance Fault Detection and Localization", https://doi.org/10.57745/KRYCYY, Recherche Data Gouv, V1



## Graphical Abstract

![img](https://github.com/mike2016/PD-CAEE/blob/main/g_abstract.pdf)

## Abstract 
High impedance faults (HIFs) present significant safety hazards such as fires and electric shock accidents, urging the development of reliable HIF detection approaches. Unfortunately, conventional protection relays often fail to detect HIFs due to their low fault currents. The prevalence of $\mu$-PMUs in distribution grids enables the monitoring of node signals, offering a promising way for accurate HIF detection. Nevertheless, challenges persist due to the scarcity of faulty data and the high costs associated with data collection, hindering further advancements in HIF detection methodologies. To tackle this challenge, we propose a hybrid artificial intelligence approach that combines the unsupervised learning capabilities of convolutional auto-encoders with the reliable physical knowledge of voltage waveform. This approach involves the transformation of node voltage signal into a phase diagram curve, leveraging the ellipse characteristics of th curve to normalize the neural network learning process. Without the need for faulty data, the proposed solution is easily applicable to other distribution grids with varying structures. Performance evaluation of the proposed approach, conducted using the IEEE-34 nodes test feeder, demonstrates its effectiveness in achieving highly accurate HIF detection across diverse faults parameter conditions and locations. Comparative analysis against other unsupervised HIF detection methods reveals the superiority of our approach, which exhibits higher detection accuracy and strong robustness to the system’s switching events.





