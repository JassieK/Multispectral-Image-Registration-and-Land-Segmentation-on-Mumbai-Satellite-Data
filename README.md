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

## 🛠️ Problems We Were Trying to Solve

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

---

This project not only showcases a typical remote sensing pipeline but also focuses on solving real-world problems that arise when working with satellite imagery, particularly in urban settings like Mumbai. By combining **image fusion**, **enhancement**, and **segmentation**, I was able to derive valuable insights from the data that could be applied to **urban monitoring**, **resource allocation**, and **environmental studies**.

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

## 🗂️ Image Flow Summary

> _(Each stage saves an image used by the next step)_

1. **Preprocessing**:  
   - 📤 Saves → `4j_Multi.jpg`, `4j_pan.jpg`
2. **Registration**:  
   - 📤 Saves → `4j_Reg_Multi.jpg`, `4j_Reg_pan.jpg`
3. **Fusion & Enhancement**:  
   - 📤 Saves → `4j_fused.jpg`, `4j_fused_enhanced.jpg`
4. **Segmentation**:  
   - 📤 Final → `4j_segmented.jpg`

---

## 📸 Suggested Image Inserts (replace these with actual outputs):

| Step | Image to Add |
|------|--------------|
| Input Stack | `4j_Multi.jpg` and `4j_pan.jpg` |
| After Registration | `4j_Reg_Multi.jpg` |
| Fused Output | `4j_fused.jpg` |
| Enhanced Output | `4j_fused_enhanced.jpg` |
| Final Segmentation | `4j_segmented.jpg` |

---

## 🚀 How to Run

You can run each `.py` file in **order** either via:
- **Jupyter Notebook cells** (recommended for visualization), or
- Command line (e.g., `python 2_register_with_sift.py`)

Make sure your images are organized in:  
`D:/Data/` → with subfolders like `1j/`, `2j/`, etc.

---

## 📝 Note

This was my **first-ever project** after starting with Machine Learning and Computer Vision, so I’ve kept things raw and genuine — the way I learned them. ❤️
