---
layout: post
comments: true
title: Image Retrieval
author: Vikram Chilkunda, Azad, Aral
date: 2024-01-01
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}
# Content-Based Image Retrieval: A Deep Dive

**Authors: Aral, Azad, Vikram**

---

## Introduction
Content-Based Image Retrieval (CBIR) leverages mathematical representations of image content to identify visually similar images within large datasets. The process involves mapping an image, represented as a matrix of pixel intensities, into a high-dimensional feature vector space. Let $$ \Phi $$ represent the feature extraction function, mapping an input image $$ I \in \mathbb{R}^{C \times W \times H} $$ (channels, width, height) to a feature vector $$ \mathbf{v} \in \mathbb{R}^D $$:

$$
\mathbf{v} = \Phi(I), \quad \text{where } \Phi : \mathbb{R}^{C \times W \times H} \to \mathbb{R}^D.
$$

Then, we input an image to the system with the goal of retrieving similar images. To do so, the system computes a similarity score between the query image's feature vector and those in the database using distance metrics, such as cosine similarity, Euclidean distance, or a variety of others not displayed here:

$$
d(\mathbf{v}_q, \mathbf{v}_i) = \| \mathbf{v}_q - \mathbf{v}_i \|_2, \quad \text{or } \frac{\mathbf{v}_q \cdot \mathbf{v}_i}{\| \mathbf{v}_q \| \| \mathbf{v}_i \|}.
$$

The top-\(k\) images with the highest similarity scores are returned as results, offering both visual and semantic relevance.

<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center;">

    <!-- <div style="text-align: center; margin: 10px;"> -->
        <img src="{{ '/assets/images/38/eiffel1.png' | relative_url }}" alt="YOLO" style="height: 200px; max-width: 100%;"/>
        <!-- <p><em>Fig 1. An example of a query image that could be used as input.</em></p> -->
    <!-- </div> -->

    <!-- <div style="text-align: center; margin: 10px;"> -->
        <img src="{{ '/assets/images/38/eiffel2.png' | relative_url }}" alt="YOLO" style="height: 200px; max-width: 100%;"/>
        <!-- <p><em>Fig 2. One of the results that may be retrieved as a result of the query image being used as input.</em></p> -->
    <!-- </div> -->

    <!-- <div style="text-align: center; margin: 10px;"> -->
        <img src="{{ '/assets/images/38/eiffel3.png' | relative_url }}" alt="YOLO" style="height: 200px; max-width: 100%;"/>
        <!-- <p><em>Fig 3. One of the results that may be retrieved as a result of the query image being used as input.</em></p> -->
    <!-- </div> -->
    <p><em>A query image on the left, along with two possible images retrieved on the right.</em></p>

</div>




---

## Real-World Applications
1. **Search Engines**: Google and Bing offer "Search by Image" functionalities.
2. **E-Commerce**: Platforms like Amazon and eBay enable product searches using images.

---

## Approaches to Image Retrieval
CBIR systems utilize various strategies to identify similar images:

## 1. Basic Feature Extraction with CNNs
- **Process:** Convolutional Neural Networks (CNNs) extract deep features like edges, textures, shapes, and spatial patterns.
- **Feature Vectors:** High-dimensional vectors (e.g., 512 or 1024 dimensions) are generated from intermediate CNN layers to represent an image.
**Example Feature Vector from ResNet-18:**
```python
[0.45, -0.12, 0.34, ..., 0.89]  # Image 1 (dog)
[0.47, -0.10, 0.31, ..., 0.87]  # Image 2 (dog)
```

## 2: LoFTR: Detector-Free Local Feature Matching with Transformers

---

### Motivation  
LoFTR introduces a new approach to **image feature detection** to address challenges faced by traditional methods, such as:  
- Poor repeatability in **low-texture areas**.  
- Errors in **repetitive patterns**.  
- Variations in **viewpoint and lighting**.  

LoFTR utilizes a **detector-free dense matching pipeline with Transformers**, diverging from traditional local feature-matching methods that rely on sequential steps: **feature detection, description, and matching**.  
Instead, LoFTR:  
1. Produces **dense matches** in regions where traditional feature detectors struggle.  
2. Refines these matches at the coarse dense level to resolve ambiguities.

---

### Techniques  
The researchers trained two specific models for different environments:  
- **Indoor Model**: Trained on the **ScanNet** dataset.  
- **Outdoor Model**: Trained on the **MegaDepth** public dataset.

---

### Architecture  
The LoFTR architecture comprises **four key components** to achieve robust local feature matching:  

1. **Feature Extraction Backbone**:  
   - Uses a standard **Convolutional Neural Network (CNN)** to extract **multi-level features** from input images.  
   - Generates **coarse-level** and **fine-level features**.  
   - **Downsampling** in CNN reduces input length, lowering computational costs.  

2. **Coarse-Level Feature Transformation**:  
   - Reshapes features into **1D vectors** and adds **positional encodings**.  
   - Processes features through a **Transformer module**, combining:  
     - **Self-Attention Layers**: For capturing global context.  
     - **Cross-Attention Layers**: For position-aware descriptors.  

3. **Matching Module**:  
   - Uses a **differentiable matching layer** to compute a **confidence matrix**, identifying likely correspondences between the feature maps of two images.  
   - Matches are selected based on:  
     - **Mutual-Nearest-Neighbor Criteria**.  
     - A **confidence threshold**.  
   - This forms the initial set of **coarse-level matches**.  

4. **Fine-Level Refinement Module**:  
   - Crops a small window around each coarse match.  
   - Refines these matches to produce **final matches with sub-pixel accuracy**.  

---

### Results  
LoFTR was evaluated in both **indoor** and **outdoor** environments, outperforming existing methods:  

- Compared to **SuperGlue** and **DRC-Net**, LoFTR produced:  
  - **More correct matches**.  
  - **Fewer mismatches**.  
- Delivered **high-quality matches** in challenging areas, such as:  
  - Texture-less walls.  
  - Floors with repetitive patterns.  
- Achieved **first-place rankings** on two public benchmarks for visual localization.  

LoFTR’s results demonstrate its effectiveness in **real-world applications** and its reliability in addressing areas where traditional methods struggle.  

---

**Note**: Red lines represent epipolar line errors greater than \(5 \times 10^{-4}\).  

Attached here is our codebase implementation in [Google Colab](https://colab.research.google.com/drive/1n5wQDbNSAakrK6i3ortlu9Cm78EA8lBN?usp=sharing).

### Implementation and Results

Let's demonstrate our implementation using two photos of a desktop setup taken from slightly different angles:

<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 20px; margin-bottom: 20px;">
    <div style="flex: 1; min-width: 300px; max-width: 400px;">
        <img src="{{ '/assets/images/38/test1.jpeg' | relative_url }}" alt="Desktop Setup View 1" style="width: 100%;"/>
        <p style="text-align: center;"><em>View 1 of the desktop setup</em></p>
    </div>
    <div style="flex: 1; min-width: 300px; max-width: 400px;">
        <img src="{{ '/assets/images/38/test2.jpeg' | relative_url }}" alt="Desktop Setup View 2" style="width: 100%;"/>
        <p style="text-align: center;"><em>View 2 of the desktop setup</em></p>
    </div>
</div>

Here's the code to run LoFTR on these images:

```python
# Configure environment and download LoFTR code
!pip install torch einops yacs kornia
!git clone https://github.com/zju3dv/LoFTR --depth 1

# Load and preprocess images
img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (640, 480))
img1_raw = cv2.resize(img1_raw, (640, 480))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference with LoFTR
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
```

When we ran this code on a pair of desktop setup images taken from slightly different angles, LoFTR was able to find 874 matching points between the images, demonstrating its robustness to viewpoint changes:

![LoFTR Matches]({{ '/assets/images/38/LoFTR-output.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4. LoFTR matching points between two images of a desktop setup, showing 874 matches across various features including the laptop, monitor, and desk surface.*

The colored lines indicate matched points between the two images, with different colors representing the confidence levels of the matches. As we can see, LoFTR successfully identified corresponding points across various features in the scene, including:
- The laptop screen and keyboard
- The external monitor displaying a mountain scene
- The desk surface and its reflections
- The window blinds in the background

This demonstrates LoFTR's ability to handle:
1. Different viewing angles
2. Varied lighting conditions
3. Reflective surfaces
4. Complex indoor environments with multiple objects

## 3: Learning Super-Features for Image Retrieval

---

### Motivation  
Traditional feature detection and matching methods, such as using **Convolutional Neural Network (CNN)** features, have significant limitations. Methods like the **Scale-Invariant Feature Transform (SIFT)** detect keypoints individually and compute descriptors around these separate, independent keypoints. This approach **ignores the spatial relationships** between keypoints, making the method:  
- Sensitive to **viewpoint changes**.  
- Sensitive to **lighting variations**.  

Even modern deep learning methods, such as **CNN-based approaches**, fail to consider spatial relationships. However, clusters of keypoints often contain **rich contextual information**. For example, a cluster of keypoints around a **building edge** has a predictable structure that can be recognized.

---

### Techniques  
To address these challenges and capture **Super-Features**, the paper introduces an **iterative Local Feature Integration Transformer**:  
- This method adapts **pre-existing attention modules** for image retrieval tasks.  
- During training, the loss is directly applied to **Super-Features**, and no additional labels or annotations are required beyond the image data itself.

<div style='display: flex; flex-direction: column; justify-content: center'>
    <img src="{{ '/assets/images/38/firetraining.png' | relative_url }}" alt="YOLO" style="height: 200px; max-width: 100%;"/>
    <p style='text-align: center'><em>Figure x: The training process for the Local Feature Integration Transformer (FIT)[2]</em></p>
</div>
---

### Architecture  
The proposed framework follows a multi-step process to generate **Super-Features**:  

1. **Feature Extraction Backbone**:  
   - A backbone **CNN** extracts dense **feature maps** that encode **local patterns** and textures from the input image.

2. **Super-Feature Extraction**:  
   - The framework detects **salient regions** from the feature maps. These regions capture **spatial and geometric context** that is meaningful for retrieval.  

3. **Context-Aware Processing**:  
   - The framework uses **non-local attention mechanisms** to incorporate spatial relationships across the image.  
   - These attention modules allow the model to focus on **non-adjacent regions** in the image, ensuring that features capture long-range dependencies.

4. **Training Objective**:  
   The network minimizes losses for both:  
   - **Keypoint Detection**: Ensures that detected keypoints are repeatable across transformed images.  
     - **Loss Function**: Cross-Entropy Loss.  
   - **Descriptor Matching**: Ensures that corresponding keypoints in two images have similar descriptors.  
     - **Loss Function**: Triplet Loss.  

---

### How It Works  
1. **Input**: A reference image from the dataset and a query image are fed into the network.  
2. **Feature Map Generation**: Dense feature maps are extracted using the backbone CNN.  
3. **Keypoint and Descriptor Extraction**:  
   - The framework detects keypoints and computes context-aware descriptors from the feature maps.  
4. **Matching**: These context-rich features enable better comparison between images, resulting in more accurate **image retrieval**.

---

### Results  
The paper evaluates the proposed method, **FIRe** (Feature Integration-based Retrieval), against state-of-the-art techniques. The network is trained on the **SfM-120k** dataset. Key results include:  

- **Memory Efficiency**:  
   - Other methods required a minimum of **7.9 GB** for the full R1M image set.  
   - The FIRe method used only **6.4 GB**.  

- **Accuracy Improvements**:  
   - On the **ROxford + R1M hard sets**, FIRe achieved an improvement of **3.3 points**.  
   - On the **RParis + R1M hard sets**, the improvement was as high as **12.2 points**.  

These results demonstrate that FIRe outperforms existing methods both in terms of **accuracy** and **memory efficiency**.

---

## Beyond Images: Feature Extraction in Genomics

While we've focused on image retrieval, the concepts of feature extraction and similarity search can be applied to many other domains. Here, we demonstrate an interesting application in genomics using the same fundamental principles. You can find our implementation in [Google Colab](https://colab.research.google.com/drive/1aBUVJpJSDyOnIqkZt4nwkAKrSr8-1sr3?usp=sharing).

Just as CNNs can extract meaningful features from images, we can use specialized models like DNABERT-2 to extract features from DNA sequences. Here's our implementation of a DNA sequence similarity search:

```python
def mystery_function(read):
    # Convert DNA sequence to numerical representation
    inputs = tokenizer([read], return_tensors='pt', padding=True)["input_ids"]
    
    # Get sequence embeddings from DNABERT-2
    hidden_states = model(inputs)[0]
    
    # Create a single vector representation via mean pooling
    read_representation = torch.mean(hidden_states, dim=1)
    
    # Calculate similarity with database sequences
    similarities = cosine_similarity(read_representation.detach().numpy(), 
                                  embedding_mean.detach().numpy())
    
    # Find and return top 5 most similar sequences
    top_indices = similarities.argsort()[0][-5:][::-1]
    for index in top_indices:
        print(dna_sequence_list[index], "similarity score:", similarities[0][index])
```

When we query this system with a DNA sequence 'ACAGCTCTCCCC', we get results like:

```
CGGCTAGGGATCGAACTCCGCGCGAGTGCC similarity score: 0.9912246
TCTGTGTTTGTTGAGTCTCCTGAGACTCCC similarity score: 0.98853076
TTAAACAGGTGGGTTCTATAGGTCTTACAT similarity score: 0.9881238
TGCCCCGGTGTAGAACGATCCGTGCACGCG similarity score: 0.98796815
CAGGTTAGACGGAGGTGCCGGTTTCCAGGG similarity score: 0.98769104
```

This demonstrates several interesting parallels with image retrieval:

1. **Feature Extraction**: Just as CNNs extract features from images, DNABERT-2 extracts features from DNA sequences by converting them into high-dimensional vectors (embeddings).

2. **Similarity Metrics**: We use cosine similarity to compare DNA sequence embeddings, similar to how we compare image feature vectors in CBIR systems.

3. **Nearest Neighbor Search**: The system returns the most similar sequences from a database, analogous to how image retrieval systems return visually similar images.

The high similarity scores (all above 0.97) suggest that our model captures meaningful patterns in the DNA sequences, allowing us to find similar genetic patterns just as we find similar visual patterns in images.

## Main Content
Your survey stafdsafsdrts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
