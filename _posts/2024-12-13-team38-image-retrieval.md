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

### 1. Feature Extraction with CNNs
- **Process:** Convolutional Neural Networks (CNNs) extract deep features like edges, textures, shapes, and spatial patterns.
- **Feature Vectors:** High-dimensional vectors (e.g., 512 or 1024 dimensions) are generated from intermediate CNN layers to represent an image.

**Example Feature Vector from ResNet-18:**
```python
[0.45, -0.12, 0.34, ..., 0.89]  # Image 1 (dog)
[0.47, -0.10, 0.31, ..., 0.87]  # Image 2 (dog)

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
