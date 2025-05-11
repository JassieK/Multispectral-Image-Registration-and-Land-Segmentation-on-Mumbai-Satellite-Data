# 🌆 MumbaiVision: Multispectral Image Fusion & Urban Segmentation

This project demonstrates a complete pipeline for fusing **multispectral and panchromatic satellite imagery** of Mumbai and segmenting urban land regions using **KMeans clustering**. It includes image preprocessing, registration with SIFT, pansharpening (HSI-based fusion), enhancement, and segmentation.

![Status](https://img.shields.io/badge/status-Completed-brightgreen)
![20250512_0156_MumbaiVision Satellite Imagery_simple_compose_01jv0gbm3je5atp1v3xbmmxr02](https://github.com/user-attachments/assets/46425642-6c2d-43d8-a13c-0b84a224dcb4)

## 🗃️ About the Data

The dataset used in this project was generously provided by my **undergraduate thesis supervisor**, a former **scientist at DRDO (Defence Research and Development Organisation)**. This was my **very first hands-on project** with satellite image analysis during my final year of engineering.

The original raw satellite data (8 GB+) was initially in high-resolution format. To enable faster prototyping and lower memory load (especially for student systems), I used JPEG conversions for the first iteration. This trade-off helped me build an end-to-end pipeline for learning and demonstration. However, the workflow can easily be extended to work with raw formats in future applications. 

Each folder (e.g., `1j`, `2j`, `3j`, `4j`) contains:
- 3-band **multispectral images** (Blue, Green, NIR)
- 1-band **panchromatic image**

All images follow the naming pattern:  
`mum_<band>_cropped_<set_number>.jpg`  
where `<band>` = `blu`, `grn`, `nir`, or `pan`.

> 🔓 **I will be uploading these data folders too, so feel free to explore or test the code with them!**

These images represent real urban areas in Mumbai and allow for a rich analysis pipeline involving image fusion, enhancement, and segmentation — just as you'd see in remote sensing or defense-grade research.

---
The four Images below represents the raw images from set number 4 :
  - Blue Image ![mum_blu_cropped_4](https://github.com/user-attachments/assets/3f9690be-3624-4a80-80c2-5ee0c3a56bf8)
  - Green Image ![mum_grn_cropped_4](https://github.com/user-attachments/assets/209a560c-496d-4567-a08c-7a12157020ca)
  - Near-Infrared Image ![mum_nir_cropped_4](https://github.com/user-attachments/assets/095a5142-df19-4909-878a-0aa2e6d268dd)
  - Panchromatic (A screenshot as the original file was too big to insert in the readme file) ![image](https://github.com/user-attachments/assets/d2702771-1980-42b4-87f7-64a7f675c562)
---

## 🛠️ Objective

This project addresses several key challenges in **satellite image processing** and **urban segmentation** using **multispectral** and **panchromatic** satellite imagery. Specifically, aimed to solve the following:

### 1. **Data Fusion for High-Resolution Imagery**
   - **Problem**: Satellite images often contain multiple spectral bands (e.g., blue, green, red, and infrared), but **panchromatic** images, while offering higher resolution, only capture intensity (grayscale). The challenge was to **combine multispectral images** with the **high-resolution panchromatic image** to create a more detailed, fused image that provides both **rich spectral information** and **high spatial resolution**.
   - **Solution**: I applied an **HSI-based fusion method** to merge the multispectral data with the high-resolution panchromatic image, improving the image's spatial resolution without losing spectral detail.

### 2. **Image Registration and Alignment**
   - **Problem**: Different satellite bands (multispectral vs. panchromatic) often need to be aligned perfectly to enable meaningful analysis. Misalignment can lead to **incorrect analysis** and **poor fusion** results.
   - **Solution**: Using **SIFT feature detection** and **RANSAC** (a robust estimation method), I successfully aligned the multispectral and panchromatic images, ensuring that the final fused image accurately represents the real-world features.

### 3. **Enhancement of Image Quality**
   - **Problem**: The initial fused images often suffer from low contrast and insufficient detail, making it difficult to distinguish subtle features in the landscape (especially for segmentation).
   - **Solution**: I applied **sharpening** and **morphological operations** (such as dilation and erosion) to enhance the fused images, making urban areas and specific features stand out more clearly.

### 4. **Urban Land Segmentation for Resource Management**
   - **Problem**: In urban planning and **resource management**, understanding land usage and classification is crucial. Traditional methods struggle with distinguishing different types of land cover (e.g., urban, vegetation, water) in large cities from satellite images.
   - **Solution**: I used **KMeans clustering** to segment the final enhanced image, grouping pixels into clusters corresponding to **urban areas**, **vegetation**, and **other land features**, providing a clear **land use classification** that can be used for urban planning or environmental analysis.


This project not only showcases a typical remote sensing pipeline but also focuses on solving real-world problems that arise when working with satellite imagery, particularly in urban settings like Mumbai. By combining **image fusion**, **enhancement**, and **segmentation**, I was able to derive valuable insights from the data that could be applied to **urban monitoring**, **resource allocation**, and **environmental studies**.

---
## 🖥️ System Design Architecture

The system architecture for this project follows a structured pipeline that includes several key steps in the satellite image analysis process. Below is the high-level breakdown of the system design, along with the individual components involved at each stage:

### 1. **Data Acquisition and Preprocessing**
   - **Input**: Raw satellite data in **multispectral (GB + NIR)** and **panchromatic (grayscale)** formats.
   - **Component**: Data is stored in organized folders, named by the set number (e.g., `1j`, `2j`, `3j`, `4j`), and each folder contains the images of different spectral bands (`blu`, `grn`, `nir`, `pan`).
   - **Process**: 
     - The images are **loaded into memory** using Python's **PIL** library.
     - Each image is **cropped** to a region of interest (ROI) and prepared for further processing.
   
   Preprocessed and merged Multispectral image ![4j_Multi](https://github.com/user-attachments/assets/02008739-5b48-41f0-a03d-a6d2e765ed7d), Preprocessed and resized Panchromatic Image ![4j_pan](https://github.com/user-attachments/assets/83fff0a5-1f6c-4cb1-a981-c5f7e134ecc6)

### 2. **Image Registration**
   - **Input**: The resized panchromatic image and the multispectral image.
   - **Component**: The **registration module** uses **SIFT (Scale-Invariant Feature Transform)** and **RANSAC** (Random Sample Consensus) to align the multispectral and panchromatic images.
   - **Process**:
     - **Keypoints and descriptors** are extracted from both images.
     - **Matching keypoints** are used to compute a transformation matrix.
     - **Perspective transformation** is applied to align the multispectral image with the panchromatic image, ensuring accurate fusion.
    
 Registered Multispectral Image ![4j_Reg_Multi](https://github.com/user-attachments/assets/5ba0feb8-4061-4076-9c73-29d396ad79cd)
 Registered Panchromatic Image![4j_Reg_pan](https://github.com/user-attachments/assets/6d6b2660-aa62-44df-be94-a8c1ec508bda)  

### 3. **Image Fusion**
   - **Input**: The multispectral images (Blue, Green, and NIR) and the panchromatic image.
   - **Component**: The **fusion module** uses the **HSI-based fusion method** to combine the multispectral images with the panchromatic image to create a **high-resolution fused image** that preserves both the spectral and spatial characteristics of the data.
   - **Process**:
     - The multispectral images (GB + NIR) are **merged** into a single image.
     - The panchromatic image is **resized** to match the resolution of the multispectral image.
     - The fusion technique enhances spatial resolution while retaining spectral information.
       
Fused Image ![4j_fused](https://github.com/user-attachments/assets/3f08c134-493d-4d27-b701-23d5b23fdd7f)

### 4. **Image Enhancement**
   - **Input**: The fused image.
   - **Component**: Image enhancement is performed using sharpening filters and morphological operations (dilation and erosion).
   - **Process**:
     - A **sharpening kernel** is applied to the fused image to enhance fine details.
     - **Morphological operations** (dilation and erosion) are applied to enhance structural features of the image.

Enhanced Fused Image ![4j_fused_enhanced](https://github.com/user-attachments/assets/65929170-57db-494e-aed0-31bcbfa471bb)    
   
### 5. **Image Segmentation**
   - **Input**: The enhanced and fused image.
   - **Component**: The **segmentation module** uses **KMeans clustering** to group pixels into different clusters representing distinct land types.
   - **Process**:
     - The fused image is converted into a **7-channel image** using color bands from both the multispectral and enhanced images.
     - **KMeans clustering** is applied to group pixels into distinct clusters (e.g., urban areas, vegetation, water).
     - The segmented image is displayed with each cluster visualized in different colors.

Segmented Final Image ![4j_segmented](https://github.com/user-attachments/assets/bce0b0de-bbbe-4f6b-80b2-ecd0edd74c6a)
   
### 6. **Output**
   - **Final Output**: A segmented image with clearly defined land cover classes, saved as a visual representation of the classified urban and environmental features.
   - **Result**: The segmented image provides a clear classification of urban areas, vegetation, and other land features, useful for urban planning, environmental monitoring, and resource management.
   - 


### **Data Flow Diagram (DFD)**

![Your paragraph text](https://github.com/user-attachments/assets/dc138611-98ac-48d1-9840-5f7a10a46f9d)

---
### **Components Summary**

- **Input Data**: Raw satellite images (multispectral + panchromatic)
- **Core Modules**:
  - Data Preprocessing (cropping, merging)
  - Image Fusion (HSI method)
  - Image Enhancement (sharpening, morphological ops)
  - Image Registration (SIFT + RANSAC)
  - Image Segmentation (KMeans Clustering)
- **Output**: Segmented urban and environmental areas in the final image.

This architecture ensures a modular approach to processing satellite images, enabling easy updates and adaptations to different image sources or processing techniques.

---

## 🧠 Processing Modules and Algorithms

This project is structured into multiple key computational stages, each building upon the previous, to process and extract meaningful insights from satellite images. Below is a detailed breakdown of each module and the algorithms used:

### 1. 📥 **Data Preprocessing**
   - **Objective**: Construct a multispectral image from the available data bands.
   - **Dataset Composition**:
     - **mum_blu_cropped_X.jpg** → Blue channel
     - **mum_grn_cropped_X.jpg** → Green channel
     - **mum_nir_cropped_X.jpg** → NIR channel (used as **Red**)
     - **mum_pan_cropped_X.jpg** → Panchromatic grayscale image
     - (Where X is the sample identifier: 1, 2, 3, 4)

   - **Band Construction**:
     - The multispectral image is formed by stacking:  
       **[Blue, Green, NIR] → interpreted as [B, G, R]**
     - Image alignment and preprocessing was done using Python’s `PIL`, `NumPy`, and `OpenCV` libraries.

---

### 2. 🧪 **Image Registration**
   - **Objective**: Align the multispectral image with the panchromatic image.
   - **Algorithms Used**:
     - **SIFT (Scale-Invariant Feature Transform)**:  
       Detects robust keypoints and descriptors regardless of scale and rotation.
     - **FLANN (Fast Library for Approximate Nearest Neighbors)**:  
       Efficient matcher for high-dimensional SIFT descriptors.
     - **RANSAC (Random Sample Consensus)**:  
       Filters out outliers and estimates a homography matrix for accurate warping.

   - **Output**: A registered multispectral image, now geometrically aligned to the panchromatic base image.

---

### 3. 🌈 **Image Fusion**
   - **Objective**: Combine the spatial details of the panchromatic image with the color information from the multispectral image.
   - **Method Used**:  
     - **HSI-Based Fusion** (Hue, Saturation, Intensity model):
       - Convert the multispectral RGB image to **HSV (OpenCV's version of HSI)**.
       - Replace the **V (intensity)** channel with the grayscale panchromatic image.
       - Slightly amplify the Saturation and Intensity channels to visually enhance the result.
       - Convert the fused image back to RGB space.

   - **Why HSI/HSV Fusion?**  
     Preserves the color characteristics from the multispectral image while enhancing detail from the high-res panchromatic image.

---

### 4. ✨ **Image Enhancement**
   - **Objective**: Sharpen and emphasize structural features in the fused image before segmentation.
   - **Techniques Used**:
     - **High-Pass Filter** (Sharpening kernel): Enhances fine edges and textures.
     - **Morphological Gradient** (I2 = Dilation - Erosion): Extracts boundary-like transitions useful for clustering.

   - **Tools**: OpenCV filters and morphological operations.

---

### 5. 🧩 **Image Segmentation**
   - **Objective**: Segment the image into different land cover regions using unsupervised learning.
   - **Algorithm Used**:
     - **KMeans Clustering**:
       - Input Features:  
         7 channels — RGB from **fused image**, RGB from **enhanced image**, and the **morphological edge band (I2)**.
       - Clustered into `k = 6` categories.
       - Result visualized using a custom color palette.

   - **Output**: A color-coded segmented image highlighting regions like vegetation, buildings, water bodies, etc.

---

### 💡 Summary Table

| Module             | Algorithms / Techniques Used                | Purpose                                                  |
|--------------------|---------------------------------------------|----------------------------------------------------------|
| Data Preprocessing | Band stacking: B, G, NIR (as R)             | Construct multispectral RGB image                        |
| Registration       | SIFT, FLANN, RANSAC                         | Align multi-source images                                |
| Fusion             | HSI-based image fusion                      | Combine spatial detail with color integrity              |
| Enhancement        | Sharpening kernel + Morphological Gradient | Improve feature distinction pre-segmentation             |
| Segmentation       | KMeans clustering on 7D features            | Unsupervised land cover classification                   |

---

Each algorithm was selected with consideration for this being a **first real-world remote sensing project**, ensuring a balance between computational efficiency and visual interpretability. The simplicity of the tools (OpenCV, NumPy, Scikit-learn) made it approachable while still achieving a technically rich processing pipeline.

---

### 🔧 Custom Enhancement Technique: Fine-Tuning with XOR

As part of the enhancement pipeline, I tried introducing a custom sharpening method using morphological operations and bitwise logic to better emphasize edges and subtle image textures. This step was designed to push beyond standard filters and extract more spatial information relevant for segmentation.

#### 🧠 What Was done:
- **Morphological Operations**:  
  Firstly computed the **difference between the dilated and eroded** versions of the panchromatic grayscale image. This highlights fine spatial variations and acts like a localized edge detector:
  ```python
  I2 = Dilation - Erosion
  Sharpened = Convolution with [[ 0, -1, 0 ], [-1, 5, -1], [ 0, -1, 0 ]]
  Sharpened = Sharpened | I2
  Final_Enhanced = Sharpened ^ I2
  
Why This Matters:
Traditional enhancement sometimes flattens subtle spatial differences. By creatively combining morphological differencing and XOR logic, I tried emphasizing non-obvious boundaries that proved helpful during unsupervised segmentation.

This tweak demonstrates the experimental approach—making small but meaningful adjustments based on visual inspection and results rather than relying solely on standard procedures.

---


## 📁 Project Structure

Each script builds on the previous one by saving intermediate outputs and using them in the next stage.

| File | Description |
|------|-------------|
| `1_preprocess_and_stack.py` | Loads and stacks the 3-channel multispectral image (Blue, Green, NIR) and single-channel Panchromatic image. Converts and saves them for the next step. |
| `2_register_with_sift.py` | Uses SIFT + RANSAC to align the multispectral image with the high-res panchromatic one. Saves the registered versions. |
| `3_enhance_fused_image.py` | Applies sharpening and morphological operations to enhance the fused image. |
| `4_fuse_hsi_method.py` | Performs HSI-based pansharpening: replaces intensity with the panchromatic channel. Boosts contrast slightly and saves the fused image. |
| `5_segment_with_kmeans.py` | Segments the final image into urban land clusters using KMeans on 7 features (RGB, enhanced RGB, and morphological difference). |

---

## 🧠 Techniques Used

- CLAHE enhancement
- SIFT feature detection
- Image registration (Homography with RANSAC)
- HSI fusion (Pansharpening)
- Morphological operations (Dilation - Erosion)
- KMeans clustering for segmentation

---


