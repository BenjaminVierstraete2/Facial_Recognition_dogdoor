# Automated dog door using facial recognition technology

The objective of this project is to demonstrate the feasibility of automating a dog door by using dog recognition technology to determine which dog is outside and if it is permitted to enter.

The project was realized using a Raspberry Pi model 3b+ using a camera module v2.

## Introducing test subjects

<p align="center">
  <img src="assets/introduction.png">
</p>

# How does it work?
## Face detection
The first step in recognizing a face in real-time is to detect the face itself. For this task, I selected the SSD-MobileNetV2-fpnlite model.

I used a pre-trained model from the TensorFlow model garden to transfer learn on data I labeled using the Visual Object Tagging Tool (VoTT). 

The original model was trained on the COCO 2017 dataset.
The data used for transfer learning is a mix of the CatsVsDogs dataset and Animalfaces dataset, both linked below in sources.

the model config can be found: https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config 


### SSD-MobileNetV2-fpnlite
this model consist of 3 parts
* MobileNet-v2 as the backbone network used for feature extraction
* SSD head (Single Shot Detection) for detection
* FPN-Lite (Feature Pyramid Network) to combine the output of the MobileNet v2 and SSD layers


### MobileNet-v2
MobileNet is a light-weight neural network architecture designed for mobile and embedded devices. It is designed to be efficient in terms of memory and computational resources, making it well-suited for use on devices with limited resources.
It is used as the base network of the model to extract features

The fully connected layer used for classification is removed, so we have a model that can extract features but leaves the spatial structure of the image.

#### Why MobileNetV2?
MobileNet V2 is designed to be lightweight and efficient, with a smaller number of parameters and lower computational requirements. Additionally, MobileNet uses depthwise separable convolutions, which can be implemented more efficiently on mobile and embedded devices.

These depthwise seperable convolutions are the middel layer of the Bottleneck Residual Blocks used in Mobilenet V2
The first layer is the expansion layer, a 1x1 convolution layer with the purpose to expand the number of channels in the data before moving on too depthwise convolution (explained below) and finally to the Bottleneck layer, this layer makes the number of channels smaller. 
As the name suggest this block makes use of a residual connection to help with the flow of gradients through the network.
More on residual blocks in the face recognition explanation.

#### Depthwise separable convolutions 
A standard convolutional layer applies a set of filters to the input, where each filter is responsible for detecting a specific feature in the input. In a depthwise separable convolution, the filters are applied separately to each channel of the input, rather than to the whole input. This results in a reduction in the number of parameters that need to be learned, and thus a decrease in computational cost.

### SSD
The SSD head is a set of one or more convolutional layers added to the model to predict bounding boxes instead of classifying. It employs a set of predefined bounding boxes, referred to as anchor boxes, to predict the location and class of objects in an input image. The SSD head combines the predictions from these anchor boxes with a non-maximum suppression (NMS) algorithm to produce the final detection results. This method of detection takes the extracted features and uses them to identify the location of objects in the image.

The Single Shot Detector (SSD) works by:

* Breaking down the input image into a grid of cells and running object detection on each cell. If no objects are detected, the cell is classified as the background class.
* Using default bounding boxes, called anchor boxes, to predict the location and class of objects within each cell. During training, the algorithm matches the appropriate anchor box with the bounding boxes of each ground truth object within an image. The anchor box with the highest degree of overlap with an object is used to predict that object's class and location.


![Example ssd](assets/ssd.png)
> Image source: MobileNet version 2 by Matthijs Hollemans https://machinethink.net/blog/mobilenet-v2/

### FPN-lite

FPN Lite is a lightweight version of the Feature Pyramid Network (FPN) architecture.
The FPN Lite module is used to combine the output of the MobileNet v2 and SSD layers, by combining features from multiple scales, the FPN Lite module provides the object detector with additional contextual information that can help improve the accuracy of the object detections.

#### Example:
<p align="center">
  <img src="assets/mobilenet_example.png">
</p>



## Face recognition 

For recognising which dog face was detected i used a CNN network implementing techniques from FaceNet.
the techniques used are:
* Triplet loss
* Residual blocks (ResNet)

## Triplet loss
The idea behind triplet loss is to train the network to produce embeddings such that similar examples are close together in the embedding space, while dissimilar examples are far apart.
The triplet loss is calculated using three inputs: an anchor image, a positive image, and a negative image. Anchor and positive are selected from the same class, while negative is selected from a different class. The goal of the network is to minimize the distance between images from the same class and maximize the distance with other classes.

<p align="center">
  <img src="assets/tripletloss.gif">
</p>

> Image Source: Face Recognition with FaceNet and MTCNN by Luka Dulčić https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/

### Online triplet generation
As outlined in the FaceNet paper (https://arxiv.org/abs/1503.03832), if faces are mislabeled or poorly imaged, they can dominate the hard positives and negatives. Despite my data being limited and well-labeled, I employed online triplet generation to select hard positive and negative triplets within a mini-batch. Hard triplets are defined as those where the positive image is closer to the negative image than to the anchor image. This technique helps to improve the robustness of the model against poorly imaged or mislabeled faces.

## Residual blocks
In a standard neural network architecture, the output of one layer is passed as input to the next layer in the sequence. However, in a network with residual blocks, the output of a layer is passed not only to the next layer but also to one or more layers further down the chain, bypassing one or more intermediate layers. This allows for the flow of information to be retained and reused throughout the network, rather than being lost or distorted as it passes through multiple layers.

why use these?
to help alleviate the problem of vanishing gradients. when training networks with many layers the gradients of the parameters with respect to the loss become very small, making it difficult for the network to learn.
Residual blocks address this problem by allowing the gradients to flow more easily through the network by using a residual connection for the gradients to flow through which bypasses one or more layers and allows the input to be added directly to the output of a layer several layers deeper in the network. This helps to retain information and gradients throughout the network.

### Example Residual block in my model:
<p align="center">
  <img src="assets/resblock.png">
</p>

> Find full model: https://github.com/BenjaminVierstraete2/Facial_Recognition_dogdoor/blob/main/FaceRecognition/models/model_plot.png

### Predicting:

Finally, the output of the model is a vector storing 128 embeddings. These embeddings are values assigned to the face. A database was made for each dog using these embeddings. This is used to predict new embeddings by measuring the minimum distance found when comparing to the database. When testing the model without a threshold, it scored 100% on the limited data of 33 test images. However, the score dropped to about 88% when introducing a minimum threshold of 0.5, meaning the dog is only classified if the minimum distance found is smaller then the threshold given.


#### Example predictions:
<p align="center">
  <img src="assets/predictions_face.png">
</p>

# Tensorflow lite

Both models were converted into tensorflow lite models.
TensorFlow Lite is a lightweight version of TensorFlow that is specifically designed for mobile and embedded devices. It is a good choice for running machine learning models on a Raspberry Pi, as it is optimized for low-power and resource-constrained devices. 
TensorFlow Lite uses a smaller, more efficient binary format for model storage.

#### Benchmark test on Tensorflow (left) and Tensorflow lite (right) in Interference speed (ms)
<p align="center">
  <img src="assets/benchmarkTfvsTflite.png">
</p>

> Image Source: Benchmarking TensorFlow and TensorFlow Lite on the Raspberry Pi by Alasdair Allan https://www.hackster.io/news/benchmarking-tensorflow-and-tensorflow-lite-on-the-raspberry-pi-43f51b796796


# Autopreprocessing model

I initially made a poor choice in my object detection model for my Raspberry Pi application because I was primarily focused on recognition. After training a YOLOv7 model on the same data, I realized that while it was more accurate, it was also much more computationally expensive and difficult to convert to TensorFlow Lite, making it a less suitable choice for embedded detection in comparison to SSD-MobileNetV2. Despite this, YOLOv7 can still be useful for object detection on devices with powerful GPUs or in cloud-based applications. 

Using this trained model i developed an automatic preprocessor that generates faces for training the recognition model. By simply providing a folder with images of dogs, the preprocessor will output all detected faces into a separate folder, leaving only the moving of the good faces into a labeled folder and removing off poor images.

### Output:
<p align="center">
  <img src="assets/preprocess.png">
</p>

# Does it work?
This is a demonstration of the model in action. The frame rate for detection is approximately 0.7, which decreases when a detected face is being processed. To determine if this model can function in real-time, we must evaluate the time it takes for each detection. If the detection process takes longer than 2 seconds, it may impact the functionality of the dog door.
<p align="center">
  <img src="assets/detection_vid.gif" width="960" height="540">
</p>

The proof of concept protoype has also been tested, u can see muchu jumping through the door after many training sessions.
<p align="center">
  <img src="assets/door_jump.gif" width="768" height="522">
</p>

In the video below, you can observe the locking mechanism in action. As you can see, there are two servo arms on each side that lock the door. It is important to note that this is only a proof of concept, and in reality, stronger servos or arms would be required as any dog can jump through and knock off these arms.

<p align="center">
  <img src="assets/locking_mechanism.gif" width="480" height="760">
</p>

If u are interested in the prototype and how it is wired? See the manuals folder for more information.

# Sources

## Object detection sources


[1] ‘Welcome to the Model Garden for TensorFlow’. tensorflow, Jan. 30, 2023. Accessed: Jan. 20, 2023. [Online]. Available: https://github.com/tensorflow/models

[2] ‘MobileNet version 2’. https://machinethink.net/blog/mobilenet-v2/ (accessed Jan. 21, 2023).

[3] Evan, ‘TensorFlow Lite Object Detection on Android and Raspberry Pi’. Jan. 20, 2023. Accessed: Jan. 20, 2023. [Online]. Available: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi

[4] ‘MobileNetV2 SSD FPN’. https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/mobilenetv2-ssd-fpn (accessed Jan. 20, 2023).

[5] ‘How single-shot detector (SSD) works?’, ArcGIS API for Python. https://developers.arcgis.com/python/guide/how-ssd-works/ (accessed Jan. 23, 2023).

[6] N. Wongsirikul, ‘YOLO Model Compression via Filter Pruning for Efficient Inference on Raspberry Pi’, Medium, Nov. 30, 2021. https://medium.com/@wongsirikuln/yolo-model-compression-via-filter-pruning-for-efficient-inference-on-raspberry-pi-c8e53d995d81 (accessed Jan. 18, 2023).

[7] Pluviophile, ‘Answer to “What is the architecture of ssd_mobilenet_v2_fpnlite_640x640?”’, Data Science Stack Exchange, Dec. 07, 2022. https://datascience.stackexchange.com/a/116807 (accessed Jan. 22, 2023).

## Face recognition sources

[8] S. Sahoo, ‘Residual blocks — Building blocks of ResNet’, Medium, Sep. 26, 2022. https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec (accessed Jan. 29, 2023).

[9] G. Mougeot, D. Li, and S. Jia, ‘A Deep Learning Approach for Dog Face Verification and Recognition’, in PRICAI 2019: Trends in Artificial Intelligence, Cham, 2019, pp. 418–430. doi: 10.1007/978-3-030-29894-4_34.

[10] F. Schroff, D. Kalenichenko, and J. Philbin, ‘FaceNet: A Unified Embedding for Face Recognition and Clustering’, in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2015, pp. 815–823. doi: 10.1109/CVPR.2015.7298682.

[11] C. Miranda Orostegui, A. Navarro Luna, A. Manjarrés García, and C. A. Fajardo Ariza, ‘A Low-Cost Raspberry Pi-based System for Facial Recognition: Sistema de reconocimiento facial sin reentrenamiento para nuevos usuarios.’, Ingeniería y Ciencia, vol. 17, no. 34, pp. 77–95, Jul. 2021, doi: 10.17230/ingciencia.17.34.4.

[12] P. D. Alexander and D. J. Craighead, ‘A novel camera trapping method for individually identifying pumas by facial features’, Ecology and Evolution, vol. 12, no. 1, p. e8536, 2022, doi: 10.1002/ece3.8536.

[13] ‘Face Recognition Walkthrough--FaceNet | Pluralsight’. https://app.pluralsight.com/guides/face-recognition-walkthrough-facenet (accessed Jan. 09, 2023).

[14] ‘Face Recognition Python* Demo — OpenVINO™ documentation — Version(latest)’. https://docs.openvino.ai/latest/omz_demos_face_recognition_demo_python.html#doxid-omz-demos-face-recognition-demo-python (accessed Jan. 11, 2023).

[15] D. Kumar, ‘Introduction to FaceNet : A Unified Embedding for Face Recognition and Clustering’, Analytics Vidhya, Jun. 21, 2020. https://medium.com/analytics-vidhya/introduction-to-facenet-a-unified-embedding-for-face-recognition-and-clustering-dbdac8e6f02 (accessed Jan. 13, 2023).

[16] ‘Face Recognition With Raspberry Pi and OpenCV - Tutorial Australia’, Core Electronics. https://core-electronics.com.au/guides/raspberry-pi/face-identify-raspberry-pi/ (accessed Nov. 10, 2022).

[17] ‘Face Recognition with FaceNet and MTCNN’. https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/ (accessed Jan. 29, 2023).

[18] N. Samsudin, ‘Building a dog search engine with FaceNet’, Analytics Vidhya, Oct. 15, 2020. https://medium.com/analytics-vidhya/building-a-dog-search-engine-with-facenet-65d1ae79dd8a (accessed Jan. 09, 2023).

[19] M. Saini, ‘Train FaceNet with triplet loss for real time face recognition…’, Medium, Jul. 20, 2019. https://medium.com/@mohitsaini_54300/train-facenet-with-triplet-loss-for-real-time-face-recognition-a39e2f4472c3 (accessed Jan. 12, 2023).

[20] S. Pospielov, ‘What is the Best Facial Recognition Software to Use in 2022?’, Medium, Jul. 07, 2022. https://towardsdatascience.com/what-is-the-best-facial-recognition-software-to-use-in-2021-10f0fac51409 (accessed Nov. 09, 2022).

[21] C.-F. Wang, ‘What’s the Difference Between Haar-Feature Classifiers and Convolutional Neural Networks?’, Medium, Aug. 04, 2018. 
https://towardsdatascience.com/whats-the-difference-between-haar-feature-classifiers-and-convolutional-neural-networks-ce6828343aeb (accessed Jan. 20, 2023).

[22] O. Moindrot, ‘Triplet Loss and Online Triplet Mining in TensorFlow’, Olivier Moindrot blog, Mar. 19, 2018. https://omoindrot.github.io/triplet-loss (accessed Jan. 23, 2023).

[23] T. A. Team, ‘The Strengths & Weaknesses of Face2Vec (FaceNet) – Towards AI’. https://towardsai.net/p/computer-vision/the-strengths-weaknesses-of-face2vec-facenet, https://towardsai.net/p/computer-vision/the-strengths-weaknesses-of-face2vec-facenet (accessed Jan. 10, 2023).

[24] V. Agarwal, ‘Face Detection Models: Which to Use and Why?’, Medium, Jul. 02, 2020. https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c (accessed Jan. 9, 2023).

[25] ‘What is Face Detection? Ultimate Guide 2023 + Model Comparison’, Sep. 06, 2022. https://learnopencv.com/what-is-face-detection-the-ultimate-guide/ (accessed Jan. 9, 2023).

## Other

[26] ‘Benchmarking TensorFlow and TensorFlow Lite on the Raspberry Pi’, Hackster.io. https://www.hackster.io/news/benchmarking-tensorflow-and-tensorflow-lite-on-the-raspberry-pi-43f51b796796 (accessed Jan. 29, 2023).

[27] ‘Convert TensorFlow models | TensorFlow Lite’, TensorFlow. https://www.tensorflow.org/lite/models/convert/convert_models (accessed Jan. 22, 2023).

[28] Q-engineering, ‘Deep learning with Raspberry Pi and alternatives in 2022 - Q-engineering’. https://qengineering.eu/deep-learning-with-raspberry-pi-and-alternatives.html (accessed Nov. 09, 2022).

[29] Q-engineering, ‘Embedded vision - Q-engineering’. https://qengineering.eu/embedded-vision.html (accessed Nov. 09, 2022).


## Data

[30] ‘Cat and Dog’. https://www.kaggle.com/datasets/tongpython/cat-and-dog (accessed Jan. 09, 2023).

[31] ‘Animal Faces’. https://www.kaggle.com/datasets/andrewmvd/animal-faces (accessed Jan. 09, 2023).





