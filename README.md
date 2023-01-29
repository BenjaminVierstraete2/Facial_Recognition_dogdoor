# Automated dog door using facial recognition technology

The objective of this project is to demonstrate the feasibility of automating a dog door by using dog recognition technology to determine which dog is outside and if it is permitted to enter.

## Introducing test subjects

<p align="center">
  <img src="assets/introduction.png">
</p>

# How does it work?
## Face detection
The first step in recognizing a face in real-time is to detect the face itself. For this task, I selected the SSD-MobileNetV2-fpnlite model.

I used a pre-trained model from the TensorFlow Object Detection API to transfer learn on data I labeled using the Visual Object Tagging Tool (VoTT). 
The original model was trained on the COCO 2017 dataset.


the model config can be found: https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config 


### SSD-MobileNetV2-fpnlite
in this model 3 networks are being used 
* MobileNet-v2 as the base network
* SSD (Single Shot Detection) as the detection network
* FPN-Lite (Feature Pyramid Network) as the feature extractor


### MobileNet-v2
MobileNet is a light-weight neural network architecture designed for mobile and embedded devices. It is designed to be efficient in terms of memory and computational resources, making it well-suited for use on devices with limited resources.

## Face recognition


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

# Sources
