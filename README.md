# Hi, We are Haris and Khubaib! Welcome to our Deep Learing Project 👋

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 

# Deep Learning Project

The repository contains the following files:

* **UNetPretrained.ipynb** – Trains a UNet++ model for lung segmentation.
* **UNet.pth** – Saved weights of the trained UNet++ model.
* **SEResNet50.ipynb** – Classifies TB from chest X-rays using SEResNet50.
* **SEResNet50.pth** – Saved weights of the trained SEResNet50 model.
* **UNet\_SENet\_Pipeline.ipynb** – Combines UNet++ and SEResNet50 for segmentation-based classification.
* **SwinTransformer.ipynb** – Applies a Swin Transformer for TB classification.

Dataset folders used in the notebooks:

* **Image-Segmentation/** – X-ray images and masks for segmentation.
* **Dataset2/**, **Dataset3/**, **Dataset4/** – X-ray images for TB classification.


## Table of Contents

1. [Introduction](#introduction)
2. [Installation Requirements](#installation-requirements)
3. [Project Structure](#project-structure)
4. [Data](#data)
5. [Training and Evaluation](#training-and-visualization)
6. [Lessons](#lessons)
7. [Screenshots](#screenshots)
   
## Introduction

This project includes implementing and comparing deep learning approaches for automated analysis of chest X-ray images. The models focus on segmenting lung regions and classifying the presence of Tuberculosis (TB) using a combination of CNN-based and Transformer-based architectures. Specifically, we train the following models:

1. **SENet** – A SEResNet50-based classifier trained directly on preprocessed raw chest X-rays.
2. **UNet++** – A segmentation model that isolates lung regions for cleaner downstream classification.
3. **UNet++ + SENet** – A two-stage pipeline where UNet++ segments lung areas, followed by SENet classification on the segmented output for improved diagnostic performance.
4. **Swin Transformer** – A Vision Transformer model applied to chest X-ray classification to explore attention-based learning.

Each notebook includes detailed preprocessing, training logic, evaluation metrics (Dice, IoU, classification reports), and GradCAM++ visualizations to interpret the model’s decision-making. This project demonstrates the effectiveness of combining segmentation and classification, and explores the use of Transformers in medical imaging.
 
## Installation Requirements

To run the notebooks in this repository, you will need the following packages:

* `numpy`
* `matplotlib`
* `seaborn`
* `opencv-python`
* `pillow`
* `torch`
* `torchvision`
* `timm`
* `albumentations`
* `scikit-learn`
* `tqdm`
* `segmentation-models-pytorch`
* `grad-cam`
* `torchsummary`

You can install these packages using pip:

```bash
pip install numpy
```

```bash
pip install matplotlib
```

```bash
pip install seaborn
```

```bash
pip install opencv-python
```

```bash
pip install pillow
```

```bash
pip install torch torchvision
```

```bash
pip install timm
```

```bash
pip install albumentations
```

```bash
pip install scikit-learn
```

```bash
pip install tqdm
```

```bash
pip install segmentation-models-pytorch
```

```bash
pip install grad-cam
```

```bash
pip install torchsummary
```

After installing the required libraries, simply run the **"Imports"** cell in each notebook to begin.

Useful Links for installing Jupyter Notebook:
- https://youtube.com/watch?v=K0B2P1Zpdqs  (MacOS)
- https://www.youtube.com/watch?v=9V7AoX0TvSM (Windows)

It's recommended to run this notebook in a conda environment to avoid dependency conflicts and to ensure smooth execution.
Also, you will need a GPU to run the notebooks. It is recommended to have a Google Colab Account (perhaps multiple accounts) for this purpose.
<h4> Conda Environment Setup </h4>
<ul> 
   <li> Install conda </li>
   <li> Open a terminal/command prompt window in the assignment folder. </li>
   <li> Run the following command to create an isolated conda environment titled AI_env with the required packages installed: conda env create -f environment.yml </li>
   <li> Open or restart your Jupyter Notebook server or VSCode to select this environment as the kernel for your notebook. </li>
   <li> Verify the installation by running: conda list -n AI_env </li>
   <li> Install conda </li>
</ul>


## Project Overview

Our project was executed over a period of approximately 1.5 months and consisted of the following five main phases:

#### 1. **Dataset Collection**

In this phase, we conducted extensive research to identify suitable datasets for our project. We curated a large collection of over 10,000 chest X-ray images from three different Kaggle datasets for classification tasks. Additionally, we sourced a dataset of 704 chest X-ray images with corresponding masks to train and evaluate our image segmentation models.

#### 2. **Baseline Model**

As our baseline, we implemented a **pretrained UNet++** model for lung segmentation. This model achieved a Dice Coefficient of 96%, providing a strong foundation for downstream classification tasks.

#### 3. **Improved Release**

In this phase, we developed a **SEResNet50** model — a Squeeze-and-Excitation ResNet architecture — for TB classification. The model demonstrated excellent performance, achieving 99% classification accuracy, along with strong results across all evaluation metrics.

#### 4. **Final Release**

We introduced a **two-stage pipeline**, where the UNet++ model first segmented the lungs, followed by classification using the SEResNet50 model. This integrated approach also achieved **99% accuracy**. Additionally, we experimented with a **Swin Transformer**, exploring transformer-based architectures for medical image classification.

#### 5. **Research Paper**

We concluded the project by writing a comprehensive **research paper** detailing our motivations, dataset preparation, methodology, model architectures, results, and analysis. This document serves as a complete overview of our approach and findings.


## Data

Training Data
  - A 1200-second video recording `(training_data.mp4)` of the robot's movement within the wooden box environment. This video is captured at 30 frames per second (fps).
  - A text file  `(training_data.txt)`containing the robot's coordinates, with 30 values recorded for each second (since video is 30 fps).

* Testing Data
  - A test video `(test01.mp4)`, 60 seconds long recorded at 30 fps.
  - A test txt file `(test01.txt)` following the same format as the `training_data.txt` file.


## Training and Visualization

The entire training process alongside the maths involved is explained in detail in the jupyter notebook. 

## Lessons

An AI project such as the one implemented here, involved many challenges, including:

1. **Handling Time Series Data:**
   - **Challenge:** Working with time series data requires careful consideration of the temporal order and dependencies between observations. This can be tricky when predicting future values based on past data.
   - **Solution:** To manage this, I implemented a lookback mechanism, which involved using previous observations to predict future values. For both KNN and Regression Tree models, this allowed me to capture temporal dependencies effectively.

2. **Implementing KNN from Scratch:**
   - **Challenge:** Building the KNN algorithm from scratch without relying on libraries like scikit-learn involved creating functions for distance calculation, finding nearest neighbors, and handling ties in predictions.
   - **Solution:** I wrote custom functions for Euclidean distance and nearest neighbor selection. To handle ties, I implemented a mechanism to decrement k until a clear prediction was obtained, ensuring robust and accurate results.

3. **Evaluating Model Performance:**
   - **Challenge:** Choosing the right value of k for the KNN model and the lookback size for the Regression Tree model required extensive evaluation. The performance needed to be assessed using metrics like RMSE.
   - **Solution:** I plotted RMSE values against different k values and lookback sizes to identify the optimal parameters. This involved iterating through various values and analyzing trends to select the best-performing configurations.

4. **Handling Model Complexity and Overfitting:**
   - **Challenge:** With increasing model complexity (e.g., higher k values or longer lookback periods), there was a risk of overfitting, where the model might perform well on training data but poorly on unseen data.
   - **Solution:** I monitored performance metrics across various configurations and chose parameters that balanced model complexity and generalization. For KNN, I observed the trend in RMSE with varying k values, and for Regression Trees, I tested different lookback sizes to find the optimal trade-off.


## Screenshots
<h3> K-Nearest Neighbour (KNN) </h3>
<h4> 1. This image shows how the value of the Root-Mean-Square-Error changes for increasing values of k. Further explanation of the results of the plot are explained in detail in the Jupyter Notebook. </h4>
<img src="pic11.png" width="450px"> <br> 

<h4> 2. This image shows the trajectory of the actual path of the micro-robot along with the trajectory predicted by the KNN algorithm <strong> implemented from scratch </strong>. You can change the value of 'start_second' in the code to compare the two trajectories for different six-second sets of times.  </h4>
<img src="pic12.png" width="450px"> <br> 

<h4> 3. This image shows the trajectory of the actual path of the micro-robot along with the trajectory predicted by the KNN algorithm <strong> implemented using the scikit-learn library </strong>. You can change the value of 'start_second' in the code to compare the two trajectories for different six-second sets of times.  </h4>
<img src="pic13.png" width="450px"> <br> 

<h3> Regression Tree </h3>
<h4> 1. This image shows how the value of the Root-Mean-Square-Error changes for increasing values of Lookback Size. Further explanation of the results of the plot are explained in detail in the Jupyter Notebook. </h4>
<img src="pic21.png" width="450px"> <br> 

<h4> 2. This image shows the trajectory of the actual path of the micro-robot along with the trajectory predicted by the KNN algorithm implemented using the scikit-learn library. You can change the value of 'start_second' in the code to compare the two trajectories for different six-second sets of times. </h4>
<img src="pic22.png" width="450px"> <br> 

 
## License

[MIT](https://choosealicense.com/licenses/mit/)
