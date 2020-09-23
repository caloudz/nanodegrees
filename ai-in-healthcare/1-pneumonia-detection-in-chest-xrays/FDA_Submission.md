# FDA  Submission

**Your Name:** Claudia Dai

**Name of your Device:** PnomoreNet

## Algorithm Description 

### 1. General Information

**Intended Use Statement:**
Assist radiological diagnosis of pneumonia in non-emergency situations.

**Indications for Use:**
* Screening studies to make sure a patient does not have pneumonia
* Radiologist's workflow re-prioritization (de-prioritize review of predicted negative cases)

The device can be utilized right after the X-ray scan. The prediction results would arrive at the radiologist together with the scan before the radiologist reviews them.

**Device Limitations:**
The device should not be used as a standalone pneumonia detection tool. Radiologists should review all scans regardless.

**Clinical Impact of Performance:**
It is recommended to use the algorithm for assisting radiological diagnosis of pneumonia in non-emergency situations.
* A negative prediction (no pneumonia) is correct with a 82.95% probability
* A positive prediction (pneumonia) is correct with a 34.71% probability

Due to the high recall of the model, the predictions suit well to aid with screening studies (to make sure a patient does not have penumonia) and radiologists' worklist prioritization (reviewing predicted negative cases can be de-prioritized). All images, regardless of the prediction, should be reviewed by radiologists and not skipped.


### 2. Algorithm Design and Function

**DICOM Checking Steps:**
The algorithm performs the following checks on DICOM images:
* is examined body part the chest area?
* is the patient position either posterior/anterior (PA) or anterior/posterior (AP)?
* is the modality a digital radiography (DX)?

**Preprocessing Steps:**
The algorithm performs the following preprocessing steps on the images:
* Images are resized to 244x244 as required by the pre-trained model.
* Images are rescaled 1/255.

**CNN Architecture:**
The model is built on a pre-trained VGG16 CNN. An attention model is built on top of the pre-trained VGG16 with convolution and pooling blocks, GAP for turning pixels on and off, and rescaling with the attempt of having the model learn attention in images.

[Attention Model Summary](fig/attn-model-summary.png)

The model outputs a probability value for binary classification with a sigmoid activation function. The learnt attention can be inspected in attention maps as below.

[Attention Map](fig/attention_map.png)


### 3. Algorithm Training

**Parameters:**
* Image augmentation
    * Rescaled 1/255
    * Centered sample-wise
    * Std normalized sample-wise
    * Horizontal flips
    * Height shift range 0.05
    * Width shift range 0.05
    * Zoom range 0.15
* Batch size
    * Training: 64
    * Validation: 1024
    * Prediction: 64
* Optimizer learning rate
    * Adam 0.0001 (1e-4)
* Layers of pre-existing architecture that were frozen
    * First 17 layers of VGG 16
* Layers of pre-existing architecture that were fine-tuned
    * All dense layers of VGG16 and attention
* Layers added to pre-existing architecture
    * Batch Normalization
    * Conv2D
    * Locally Connected 2D
    * Conv2D
    * Multiply
    * Global Avg Pooling
    * Global Avg Pooling
    * RescaleGAP
    * Dropout
    * Dense
    * Dropout
    * Dense
    
[Algorithm Training Performance](fig/history-plot-vgg-attn-sep.png)

[Algorithm Training Performance](fig/history-plot-vgg-attn-tog.png)

* The training loss and accuracy show improvements, so the model is learning.
* However, training has a bit of noisy movement while validation has a lot of noisy movements, and a large gap remains between both curves which never converge (with early stopping in place). This indicates that the validation dataset might be unrepresentative.
* Another indicator is that validation loss is much higher, and accuracy much lower, so the validation data seems to be harder for the model to make predictions with.

[ROC Curve Plot](fig/auc-plot-vgg-attn.png)

The ROC curve indicates that the model has learned something from the data.

[PR by Threshold Plot](fig/prth-plot-vgg-attn.png)

**Final Threshold and Explanation:**
The maximum f1-score is 0.4793 with a threshold of 0.48. In "[Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning](https://arxiv.org/pdf/1711.05225.pdf%202017.pdf)", Rajpurkar, Pranav, et al. (2017) list a comparison of their CheXNet with the performance of radiologists:

|                  | F1-score | 95% CI    | 
|------------------|-------|--------------|
| Radiologist 1    | 0.383 |(0.309, 0.453)|
| Radiologist 2    | 0.356 |(0.282, 0.428)|
| Radiologist 3    | 0.365 |(0.291, 0.435)|
| Radiologist 4    | 0.442 |(0.390, 0.492)|
|------------------|-------|--------------|
| Radiologist Avg. | 0.387 |(0.330, 0.442)|
| CheXNet          | 0.435 |(0.387, 0.481)|
|------------------|-------|--------------|
| PnomoreNet       | 0.479 |              |

PnomoreNet's f1-score is higher than both the radiologists' average and CheXNet. Because PnomoreNet has a high recall with a low precision, the model contributes value in terms of its predicive value of negatives.


### 4. Databases

**Description of database of patient data used:**

[Patient Gender](fig/atient-gender.png)

The data contained records from 56.49% male and 43.51% female.

[Patient Age](fig/patient-age.png)

The minimum age is at 0 and the maximum age at 100 (after pruning outliers of records with age > 100. For example, there were records with patient age 414.

[View Position](fig/view-position.png)

There were 39.97% of AP view positions and 60.03% of PA view positions.

[Pneumonia Co-Occurence Other Diseases](fig/pneumonia-cooccurrence.png)

Out of all 30805 unique patients, 1008 patients have pneumonia. Out of these pneumonia patients, there are 27 unique patients who have only pneumonia and no other disease. The most common diseases that co-occur with pneumonia are: Infiltration, Edema, Effusion, and Atelectasis.

**Description of Training Dataset:** 
The training dataset consisted of 2290 image files, with a 50/50 split of positive and negative pneumonia cases.

**Description of Validation Dataset:** 
The validation dataset consisted of 1430 image files, with a 20/80 split of positive and negative pneumonia cases to approach a more realistic distribution of pneumonia in the real world.


### 5. Ground Truth

Training and validation data was sampled from a larger dataset curated by the NIH specifically to address the problem of a lack of large x-ray datasets with ground truth labels to be used in the creation of disease detection algorithms.

There are 112,120 X-ray images with disease labels from 30,805 unique patients in this dataset.  The disease labels were created using Natural Language Processing (NLP) to mine the associated radiological reports. The labels include 14 common thoracic pathologies: 
- Atelectasis 
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia 

The biggest limitation of this dataset is that image labels were NLP-extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.

The original radiology reports are not publicly available but you can find more details on the labeling process [here.](https://arxiv.org/abs/1705.02315) 

Original dataset contents: 
1. 112,120 frontal-view chest X-ray PNG images in 1024*1024 resolution
2. Meta data for all images (Data_Entry_2017.csv): Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, Patient Gender, View Position, Original Image Size and Original Image Pixel Spacing.


### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**
* Age: 0 to 100
* Gender: Female and Male
* Type of imaging modality: DX
* Body part imaged: Chest

**Ground Truth Acquisition Methodology:**
The gold standard for obtaining ground truth would be to perform either a Sputum test or Pleural fluid culture. However, these tests are expensive, and diagnosis is often informed by the report of a radiologist. As this model's intended use is to assist radiologists, images can be validated by three independent radiologists per silver standard.

**Algorithm Performance Standard:**
The performance standard for the model should be calculated with the f1-score against the silver standard. The average radiologist achieves an f1-score of 0.387. See [Rajpurkar, Pranav, et al. (2017)](https://arxiv.org/pdf/1711.05225.pdf%202017.pdf). The model's f1-score should exceed the radiologist's f1-score and statistical signficance of the improvement of the average f1-score should be taken into account when assessing model performance.
