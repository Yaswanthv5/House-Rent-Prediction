<div align="center"> <h1>House Price Prediction Model</h1>
<br>
<h2>The house price prediction model is a regression model used in real estate to predict the model.</h2>
</div>

<br>

# 🏠 House Price Prediction with XGBoost 🚀  

## 📌 Overview  
This project builds an **ML model to predict house prices** using the **California Housing Dataset**.  
- **Best Model:** `XGBoost` (after hyperparameter tuning)  
- **Performance:** R² Score = `0.8616`
- **Deployment:** Flask API with Docker  

---

## 📊 1️⃣ Dataset & Features  
We use **California Housing Data** from `sklearn.datasets`.  

| Feature Name  | Description |
|--------------|------------|
| `MedInc`  | Median income in block ($10,000s) |
| `HouseAge` | Median age of houses |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Population of the block |
| `AveOccup` | Average household occupancy |
| `Latitude` | Latitude of block |
| `Longitude` | Longitude of block |

✅ **Newly Engineered Features**:  
- `RoomsPerHousehold = AveRooms / AveOccup`  
- `BedrmsPerRoom = AveBedrms / AveRooms`  
- `PopPerHousehold = Population / AveOccup`  

---

## ⚙️ 2️⃣ Model Training Pipeline  
1. **Data Preprocessing** (Handling missing values, outliers, feature engineering).  
2. **Train & Evaluate Models**:  
   - `RandomForestRegressor` (Baseline)  
   - `GradientBoostingRegressor`  
   - `XGBoost` (Best Model ✅)  
3. **Hyperparameter Tuning** (`Optuna` for best parameters).  
4. **Logging with MLflow** (Track experiment results).  
5. **Save Model using `joblib`**.  

---

## 🚀 3️⃣ Deployment  
### **A. Run Inference Locally**  
1. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt

   python app.py
   ```

### website link hosted in Azure using docker container image
--> https://housepriceapi.azurewebsites.net/