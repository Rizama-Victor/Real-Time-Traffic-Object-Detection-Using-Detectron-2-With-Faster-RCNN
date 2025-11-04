# 🛣️ Real-time Traffic Object Detection Using Detectron 2 with Faster RCNN
This repository contains the implementation of my research titled [_"Real-time Traffic Object Detection Using Detectron 2 with Faster RCNN"_](https://wjarr.com/content/real-time-traffic-object-detection-using-detectron-2-faster-r-cnn), published in World Journal of Advanced Research and Reviews (2024) authored by [Rizama Victor](https://github.com/Rizama-Victor) and [Prince Abiamamela Obi-Obuoha](https://github.com/MelaObuoha) at the National Centre for Artificial Intelligence and Robotics (NCAIR), Nigeria.
## 🔍 Overview
Accurate identification of objects from real-time video is necessary for effective traffic analysis as this plays a vital role in providing drivers and authorities a comprehensive understanding of the road and surrounding environment. Fortunately, modern algorithms such as neural network based architecture with high detection accuracy, like Faster R-CNN are at the center of this process. The project focuses on developing a model capable of identifying key traffic objects such as traffic lights, vehicles, buses, crossroads etc., for urban traffic applications using an interactive Gradio interface and Detectron2’s Faster R-CNN architecture.

## 🎯 Research Objectives

- To develop a computer vision model capable of detecting and identifying multiple key traffic objects such as vehicles, traffic lights, buses, crosswalks, motorcycles etc.
- To design an interactive Gradio-based interface that allows users to perform live detection on video feeds or static images.
- To demonstrate the practical application of neural network–based object detection in urban traffic management systems.
- To evaluate and visualize the detection performance of the model on real-world traffic data.

## 🛠️ Tools and Technologies Used

| Tool / Library              | Purpose in the Project                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Python**                  | Served as the main programming language for implementing the entire detection pipeline.               |
| **PyTorch**                 | Provided the deep learning framework for the Faster R-CNN architecture.                      |
| **TorchVision**             | Supported loading pretrained weights, dataset handling, and image transformation utilities.           |
| **OpenCV**                  | Used for capturing, reading, and processing video and image frames for inference.                     |
| **Detectron2**              | Core framework used to configure, train, and run the Faster R-CNN model for traffic object detection. |
| **Gradio**                  | Enabled creation of an interactive user interface for real-time video and image detection testing.    |
| **Google Colab**            | Provided GPU computational resources and cloud environment for running model training and inference.  |
| **Google Drive**            | Used as storage for saving and loading the trained model weights (`model_final.pth`).                 |
| **NumPy**                   | Supported image array manipulation and data preprocessing.                                            |
| **Matplotlib**              | Used for visualization and performance graph plotting during model training.                          |
| **Model Zoo (Detectron2)**  | Provided the base Faster R-CNN configuration file used for fine-tuning the model.                     |
| **Metadata (Detectron2)**   | Allowed setting and displaying object class names during visualization.                               |
| **Visualizer (Detectron2)** | Handled drawing of detection boxes and class labels on images and frames.                             |

## 🪜 Step-by-step Procedure

## 📚 References
Obi-Obuoha A. , Rizama V.S. _"Real-time traffic object detection using detectron 2 with faster R-CNN"_ World Journal of Advanced Research and Reviews Volume 28 Issue 1 2024 Page 2173–2189.
[Access the Full Paper](https://wjarr.com/sites/default/files/WJARR-2024-3559.pdf)

## 📌 Note
Please kindly note that this README file is a summarized version of the full implementation of this research. The complete implementation can be accessed via the [program script](Real-Time-Traffic-Object-Detection-Using-Detectron-2-With-Faster-RCNN-MAIN.ipynb) and [interface implementation](Real-Time-Traffic-Object-Detection-Using-Detectron-2-With-Faster-RCNN-GRADIO_INTERFACE.ipynb). Dataset and Model Weights can be provided upon request.
