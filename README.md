# Deep Learning Based Product Reordering Prediction

## Project Overview
This project aims to predict whether a user will reorder a product based on their past purchase behavior using deep learning. The goal is to help e-commerce platforms improve **personalized recommendations, inventory management, customer retention, and marketing optimization**.  

The model is trained on the **dataset**, which contains historical orders, products, aisles, and department information.

---

## Skills I Learned
- Data preprocessing and feature engineering  
- Building and training Deep Neural Network (DNN) models  
- Classification and probability prediction  
- Evaluation using classification metrics: Precision, Recall, F1-Score, ROC-AUC  
- Working with large tabular datasets  
- Model optimization and fine-tuning  
- Preparing deployment-ready models  
- Modular coding and reproducibility practices  

---

## Dataset
The dataset includes the following CSV files:  
- `orders.csv` – contains customer order history  
- `order_products__train.csv` – products purchased in training orders  
- `order_products__prior.csv` – products purchased in prior orders  
- `products.csv` – product metadata  
- `aisles.csv` – aisle metadata  
- `departments.csv` – department metadata  

**Key variables:** `user_id`, `product_id`, `order_id`, `aisle_id`, `department_id`, `reordered`, `order_dow`, `order_hour_of_day`, `days_since_prior_order`.

---

## Approach

### 1. Data Understanding and Exploration
- Loaded and explored all datasets  
- Merged datasets to create a **complete user-product interaction table**

### 2. Feature Engineering
- **User-based features**: average days between orders, total orders  
- **Product-based features**: reorder ratio, number of times reordered  
- **User-product interaction features**: how many times a user reordered a product

### 3. Preprocessing
- Handled missing data  
- Encoded categorical variables (`aisle`, `department`)  
- Normalized numerical features for neural network input

### 4. Model Building
- Built a **Feedforward Deep Neural Network (DNN)**  
- Used Dropout layers to prevent overfitting

### 5. Training and Validation
- Split data into training and validation sets (80:20)  
- Used **Binary Cross-Entropy** as the loss function  
- Applied **EarlyStopping** to avoid overfitting

### 6. Evaluation
- Evaluated model using **Accuracy, Precision, Recall, F1-Score, ROC-AUC**  
- Plotted confusion matrix for detailed analysis

### 7. Deployment Ready
- Saved trained model as `product_reorder_model.h5`  
- Prediction pipeline ready for future use  

---

## Results
- Achieved good predictive performance on validation data  
- Confusion matrix shows the model can differentiate reordered vs non-reordered products effectively  

---

## Project Structure

```bash
project_folder/
    ├── orders.csv
    ├── order_products__train.csv
    ├── order_products__prior.csv
    ├── products.csv
    ├── aisles.csv
    ├── departments.csv
    ├── main.py
    └── README.md
```


## How to Run
1. Place all CSV files in the same folder as `main.py`  
2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the training pipeline:
    ```bash
    python main.py
    ```
4. The trained model will be saved as **product_reorder_model.h5**

### Conclusion

This project demonstrates how deep learning can be applied to predict user reorder behavior, which is useful for **personalized recommendations**, **inventory forecasting**, and **targeted marketing** campaigns in **e-commerce**.

By using a **modular**, **reproducible**, and **PEP8-compliant** pipeline, this project is maintainable and easy to extend for future improvements.
