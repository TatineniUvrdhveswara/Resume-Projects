Refere this link to use data set-"https://www.kaggle.com/c/digit-recognizer/data?select=train.csv" 

# Digit Recognition Project

## 📌 Project Overview
This project aims to develop a **Digit Recognition Model** using machine learning techniques. The model is designed to accurately identify handwritten digits from image data. The project workflow includes:

✅ Data Collection and Preprocessing  
✅ Data Cleaning and Feature Engineering  
✅ Exploratory Data Analysis (EDA)  
✅ Model Building and Evaluation  
✅ Model Deployment for Real-Time Predictions  

## 🚀 Technologies Used
- **Python** (Jupyter Notebook)
- **Pandas**, **NumPy** (Data Processing)
- **Matplotlib**, **Seaborn** (Data Visualization)
- **scikit-learn**, **TensorFlow/Keras** (Machine Learning Models)
- **Flask/Streamlit** (Deployment)

## 📂 Dataset Information
The dataset used is the **MNIST dataset**, containing 28x28 pixel grayscale images of handwritten digits (0-9).

- **Input Features:** Pixel values representing grayscale intensity of each image.  
- **Output Feature:** Digit label (0-9)

## 🔍 Data Preprocessing
1. **Resizing and Normalization:** Resized images to uniform dimensions and scaled pixel values to a range of 0-1.  
2. **Data Augmentation:** Applied rotation, shifting, and zooming to improve model robustness.  
3. **Train-Test Split:** Divided the dataset into training and testing sets to evaluate model performance.

## 📊 Exploratory Data Analysis (EDA)
- Visualized sample images to understand digit patterns.  
- Plotted class distributions to ensure data balance.  
- Visualized pixel intensity distributions for better feature understanding.

## 🤖 Model Building and Evaluation
The following models were implemented and evaluated:

| **Model**             | **Accuracy** | **Loss**      |
|-----------------------|---------------|----------------|
| Random Forest          | 92.5%         | 0.24            |
| XGBoost                | 91.8%         | 0.27            |
| Decision Tree          | 85.4%         | 0.45            |
| K-Nearest Neighbors    | 88.9%         | 0.38            |
| CNN (Best Model)       | 97.5%         | 0.08            |

**Best Model:** Convolutional Neural Network (CNN) (Highest Accuracy and Lowest Loss)

## 🖥️ Deployment
- Deployed the best-performing model using **Flask** for web-based interaction.
- Users can upload digit images, and the model will predict the corresponding digit.

## 📋 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to train and evaluate models.
4. To deploy the model using Flask, run:
   ```bash
   python app.py
   ```


## 📈 Future Improvements
✅ Enhance model accuracy using advanced architectures like ResNet or EfficientNet.  
✅ Implement automated hyperparameter tuning.  
✅ Develop a mobile-friendly interface for real-time digit recognition.

## 🧑‍💻 Contributing
Contributions are welcome! Feel free to raise issues or submit pull requests.

## 📞 Contact
For questions or collaboration, reach out at: **tatineniuvrdhveswara@gmail.com**

