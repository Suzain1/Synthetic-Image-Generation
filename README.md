# 📌 **Synthetic Image Generation, Preprocessing, and Flux Model Forward Pass**

## 📝 **Project Overview**
This project demonstrates the pipeline for:
- ✅ Generating synthetic images using Stable Diffusion
- ✅ Preprocessing images using PIL
- ✅ Implementing a Flux-based neural network in Julia

---

## 📂 **Project Structure**

📁 **Generated Images/**  
➡️ Contains synthetic images generated via Stable Diffusion.

📁 **Preprocessed Images/**  
➡️ Stores resized, normalized, and grayscale images ready for model input.

📄 **flux_model.jl**  
➡️ Flux-based neural network implementation for forward pass.

📄 **image_generation.py**  
➡️ Python script for generating images using Stable Diffusion.

📄 **image_preprocessing.py**  
➡️ Python script for resizing, normalizing, and converting images to grayscale.

---

## 🚀 **How to Run the Project**

### **1️⃣ Generate Synthetic Images**
Run the following command in Python:
```bash
python image_generation.py
```
This will create and save synthetic images in the `Generated Images` folder.

### **2️⃣ Preprocess the Images**
```bash
python image_preprocessing.py
```
This resizes, normalizes, and converts the images to grayscale, storing them in `Preprocessed Images/`.

### **3️⃣ Set Up Julia Environment**
1. Install [Julia](https://julialang.org/downloads/).
2. Open Julia and install the required packages:
```julia
using Pkg
Pkg.add(["Flux", "Images", "NPZ"])
```
3. Ensure your Julia environment is correctly set up before running the model.

### **4️⃣ Run the Flux Model in Julia**
```julia
julia flux_model.jl
```
This loads the preprocessed image, runs it through the model, and prints the output probabilities.

---
## 📊 **Expected Output**
- The Flux model outputs a probability vector (e.g., `Float32[0.1, 0.09]`).
- The sum of values should be approximately **1.0** (due to softmax activation).
- **Assumption:** The model is built for **binary classification**, meaning it has **2 output values** representing probabilities for two classes.

---

## 🛠 **Requirements**
### **🔹 Python Libraries:**
- diffusers
- OpenCV / PIL
- NumPy

### **🔹 Julia Packages:**
- Flux.jl
- Images.jl
- NPZ.jl

---


