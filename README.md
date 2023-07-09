# Breast-tissue-lesion-detection-in-ultrasound-images-with-DSMI

Breast cancer is common type of diagnosed cancer among women. Its fatality rates can
be significantly decreased with early diagnosis of abnormalities in the breast. Ultrasound
images are frequently utilised for diagnosis since they are less expensive and quicker than
other modalities. Human readers might overlook certain abnormal findings, and expert
cancer detection is still time-consuming. So, to find breast lesions, we need an automated
tool. Deep learning has also developed to the point that it can now be widely used for
object detection and image classification.

-----------------------------------------------------------------------------------------------------------------------------------------
In order to do this, we have come up with an architecture for detecting breast tumour
lesions that is based on disease-specific meta information using Deep Learning. We have
also done Preprocessing and Postprocessing to improve the model performance.
This proposed architecture can locate and diagnose breast lesions. We have used BUSI
dataset, which includes 780 images(210 malignant, 133 normal and 487 benign samples).
The test results on BUSI dataset show that the proposed method has good performance.
We have used BUSC dataset which includes 250 images(150 malignant and 100 benign
samples) to check the effect of not using Imagenet weights for pretraining purpose. Our
proposed architecture achieved 86.31% Benign accuracy, 85.71% Malignant accuracy and
86.13% Combined accuracy. The Malignant accuracy has improved by 48% and Benign
accuracy has improved by 37.14% as compared to BUSnet.
