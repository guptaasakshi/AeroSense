
# 🌍 AeroSense – Smart AQI Monitoring & Prediction System

AeroSense is a smart air quality monitoring and prediction system that analyzes air pollution data, predicts Air Quality Index (AQI), identifies pollution hotspots, discovers pollution patterns, and provides health alerts and recommendations using data mining, machine learning, and deep learning techniques.

---

## 📌 Project Overview

Air pollution is a major environmental and health issue. AeroSense is designed to help users understand current and future air quality conditions in an easy and interactive way.

This project:
- Fetches real-time AQI data using an online API  
- Analyzes historical air pollution data  
- Predicts future AQI values  
- Identifies pollution hotspot regions  
- Finds frequent pollution patterns  
- Displays health alerts and activity recommendations  

All results are shown through an interactive Streamlit dashboard.

---

## 🧠 Algorithms Used

- Decision Tree – AQI category classification  
- K-Means Clustering – Pollution hotspot detection  
- FP-Growth Algorithm – Frequent pollution pattern mining  
- LSTM (Long Short-Term Memory) – 7-day AQI forecasting  

---

## 🗂️ Dataset Description

### Historical Dataset (CSV)
- File: city_day.csv  
- Daily air pollution data of Indian cities  
- Used for ML training, clustering, forecasting  

### Real-Time Dataset (API)
- Source: AQICN – World Air Quality Index  
- Live AQI & pollutant values  

---

## 🖥️ Project Structure

AeroSense/
│
├── app.py
├── city_day.csv                  # Main historical dataset 
├── requirements.txt
│
├── models/
│   ├── lstm_model.h5
│   ├── decision_tree.pkl
│
├── notebooks/
│   ├── kmeans.ipynb
│   ├── fp_growth.ipynb
│
├── csv_files/                     # 🔹 New Folder for generated CSVs
│   ├── 7day_aqi_forecast.csv
│   ├── clustered_pollution_data.csv
│   ├── fp_growth_patterns.csv
│   ├── fp_growth_rules.csv
│   ├── kmeans_clusters.csv
│   ├── preprocessed_airquality_data.csv
│
├── screenshots/                   # Dashboard & output screenshots
│
├── README.md


---

## 🛠️ Technologies & Libraries

- Python  
- Pandas, NumPy  
- Scikit-learn  
- MLXtend  
- TensorFlow & Keras  
- Matplotlib, Plotly  
- Streamlit  
- Requests  

---

## ▶️ How to Run the Project

1. Clone repository  
git clone https://github.com/your-username/AeroSense.git

2. Go to folder  
cd AeroSense

3. Install libraries  
pip install -r requirements.txt

4. Run app  
streamlit run app.py

---

## 📊 Output

- Interactive dashboard  
- Real-time AQI display  
- Hotspot map  
- Pollution patterns  
- AQI forecast  
- Health alerts  

---

## 🎯 Applications

- Public health awareness  
- Environmental monitoring  
- Government planning  
- Academic research  

---

## 🔮 Future Scope

- Mobile app integration  
- Advanced prediction models  
- Weather data integration  
- Automated alerts  

---

## 📚 References

- CPCB: https://cpcb.nic.in  
- AQICN: https://aqicn.org  
- Scikit-learn: https://scikit-learn.org  
- TensorFlow: https://www.tensorflow.org  
- Streamlit: https://docs.streamlit.io  

---

## 👩‍💻 Author

Sakshi Gupta  
B.Tech – Computer Science & Engineering  

⭐ Star the repo if you like it!
