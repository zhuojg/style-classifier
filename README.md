# Style Classifier
This model tries to classify images according to their design style.  
Data is coming from Tongji Tezign D&A.I Lab(https://sheji.ai).

---

# *Multi-class*
This is the multi-class version of style classifier, which is capable of distinguishing modern, pop art, lively and vintage style.

## Model - class StyleClassifier
Style classifier use a pre-trained CNN model to extract style features, and then classify them using fully connected layer. Generic models for image recognition task on ImageNet such as VGG-16, VGG-19, ResNet and DenseNet are all suitable. According to [1], Gram Matrix is calculated after extracting the feature maps using CNN model, in order to represent the style feature of a design.

## Data
Data is coming from Tongji Tezign Design&A.I. Lab(https://sheji.ai). The whole dataset contains about 9000 designs and 15 tags, but not all designs and tags are suitable for this style task.  
For this multi-classifier, only 4 tags - modern, pop art, lively and vintage, are used for training and validating, corresponding to about 3000 designs. 

## Training
80% data are used for training and 20% for validation.  
Using SGD with 1e-4 learning rate and 0.1 momentum.
![image](https://raw.githubusercontent.com/zhuojg/style_classifier/master/log/loss.png)  
![image](https://raw.githubusercontent.com/zhuojg/style_classifier/master/log/acc.png)  

## Result
The difference in accuracy between the models is relatively small. The training set accuracy rate is about 90%, and the verification set accuracy rate is less than 70%. This shows a significant overfitting.
A confusion matrix on validation set can be seen at the bottom of this file.  
Pre-trained model based on VGG-16 can be downloaded from https://drive.google.com/file/d/1-1-tL1Enw_KYg4wqOyV-FFFzUmM-buGp/view?usp=sharing.
   
   
---
   
   
[1] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

