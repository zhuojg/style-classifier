# Style Classifier
This model tries to classify images according to their design style.  
Data is coming from Tongji Tezign D&A.I Lab(https://sheji.ai).

## Model
This classifier is based on VGG16.  
I use the feature maps from the last convolutional layer and calculate the gram matrix, which is supposed to be the style feature.

## Result
My training set is not big enough(about 1000 images), and I get 90% accuracy on training set and get 80% accuracy on validation set.  
The result is not good enough due to the lack of data, and there seems to be over-fitting on training set.  
Pre-trained model can be downloaded on https://drive.google.com/open?id=1-7Q8bhIAIiKSyGYLW4utC1nLxberZLzk.
