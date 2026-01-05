# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import plotly.express as px 
from mlxtend.frequent_patterns import fpgrowth, association_rules


# optional plotly
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# optional sklearn scaler
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    SKL_AVAILABLE = True
except Exception:
    SKL_AVAILABLE = False

# optional tensorflow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="AQI Prediction & Analysis Dashboard", layout="wide")
st.title("🌍 AQI Prediction & Analysis Dashboard")

# ---------------- CONFIG ----------------
AQICN_API_TOKEN = "ae56dc7583110724a47826e6d93c9274f70ac4f2"  # Replace with your token
MODEL_FILE = "decision_tree_aqi_model.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_FILE = "label_encoder.pkl"
HIST_CSV = "city_day.csv"

# ---------------- load ML artifacts ----------------
def load_artifacts():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        le = joblib.load(LABEL_FILE)
        return model, scaler, le
    except Exception as e:
        # Not fatal for the app — some features will be disabled
        st.warning("Model/scaler/label encoder could not be loaded. Some features will be disabled.")
        return None, None, None

model, scaler, le = load_artifacts()

# ---------------- helper functions ----------------
def load_history():
    try:
        df = pd.read_csv(HIST_CSV)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        numeric_cols = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene","AQI"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # treat AQI 0 as missing and forward/backfill small gaps
        if "AQI" in df.columns:
            df["AQI"] = df["AQI"].replace(0, np.nan)
            df["AQI"] = df["AQI"].ffill().bfill()
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def fetch_aqi_pollutants(city, token):
    if not token or token.strip() == "" or token == "YOUR_API_TOKEN":
        return None, "AQICN API token not set"
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    try:
        r = requests.get(url, timeout=10)
    except Exception as e:
        return None, f"Network error: {e}"
    if r.status_code != 200:
        return None, f"API returned status code {r.status_code}"
    try:
        j = r.json()
    except Exception as e:
        return None, f"JSON parse error: {e}"
    if j.get("status") != "ok":
        return None, f"API status not ok: {j.get('status')}"
    iaqi = j["data"].get("iaqi", {})
    def get_v(k):
        entry = iaqi.get(k)
        if entry is None:
            return None
        return entry.get("v")
    pollutants = {
        "PM2.5": get_v("pm25") or 0,
        "PM10": get_v("pm10") or 0,
        "NO2": get_v("no2") or 0,
        "CO": get_v("co") or 0,
        "O3": get_v("o3") or 0
    }
    if all(v is None for v in pollutants.values()):
        return None, "API returned no pollutant values"
    return pollutants, None

def predict_category_numeric(pollutants):
    default_features = ["PM2.5","PM10","NO2","CO","O3"]
    df_in = pd.DataFrame([{f: pollutants.get(f,0) for f in default_features}])

    # Convert all to numeric
    for c in df_in.columns:
        df_in[c] = pd.to_numeric(df_in[c], errors="coerce").fillna(0.0)

    # If all pollutants are zero → insufficient data
    if float(df_in.sum(axis=1).iloc[0]) == 0.0:
        return None, None, "Insufficient pollutant data"

    try:
        # ------------------------------
        # Calculate Numeric AQI manually
        # ------------------------------
        numeric_aqi = int(
            0.5 * pollutants["PM2.5"] +
            0.2 * pollutants["PM10"] +
            0.1 * pollutants["NO2"] +
            0.1 * pollutants["CO"] +
            0.1 * pollutants["O3"]
        )

        # ------------------------------
        # Categorize using Indian AQI
        # ------------------------------
        if numeric_aqi <= 50:
            pred_cat = "Good"
        elif numeric_aqi <= 100:
            pred_cat = "Satisfactory"
        elif numeric_aqi <= 200:
            pred_cat = "Moderate"
        elif numeric_aqi <= 300:
            pred_cat = "Poor"
        elif numeric_aqi <= 400:
            pred_cat = "Very Poor"
        else:
            pred_cat = "Hazardous"

        return pred_cat, numeric_aqi, None

    except Exception as e:
        return None, None, f"Model prediction error: {e}"


def ensure_latlon(df):
    if ("lat" in df.columns) and ("lon" in df.columns):
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        return df
    uniq = sorted(df["City"].unique())
    coords = {}
    base_lat, base_lon = 20.0, 75.0
    for i, city in enumerate(uniq):
        coords[city] = (base_lat+(i%10)*0.8+(i//10)*0.3, base_lon+(i%10)*0.6+(i//10)*0.2)
    df = df.copy()
    df["lat"] = df["City"].map(lambda x: coords.get(x,(base_lat,base_lon))[0])
    df["lon"] = df["City"].map(lambda x: coords.get(x,(base_lat,base_lon))[1])
    return df

def safe_marker_size(series,min_size=6):
    s = pd.to_numeric(series.fillna(0), errors="coerce").fillna(0)
    if (s==0).all():
        return np.full(len(s), min_size)
    sizes = (s-s.min())/(s.max()-s.min()+1e-6)
    sizes = np.clip(sizes*40+min_size, min_size, 80)
    return sizes

def lstm_forecast_for_city(city_df, days_ahead=7, epochs=20, verbose=0):
    if not TF_AVAILABLE:
        return None,"TensorFlow not installed"
    if city_df is None or "AQI" not in city_df.columns:
        return None,"Historical AQI data missing"
    series = city_df.sort_values("Date")["AQI"].astype(float).reset_index(drop=True)
    if len(series)<60:
        return None,f"Not enough points ({len(series)}) for LSTM, need ~60"
    if SKL_AVAILABLE:
        scaler_local = MinMaxScaler()
        scaled = scaler_local.fit_transform(series.values.reshape(-1,1))
    else:
        arr = series.values.astype(float)
        minv,maxv = arr.min(),arr.max()
        scaled = ((arr-minv)/(maxv-minv+1e-6)).reshape(-1,1)
        scaler_local = None
    seq_len=14
    X,y=[],[]
    for i in range(seq_len,len(scaled)):
        X.append(scaled[i-seq_len:i,0])
        y.append(scaled[i,0])
    X = np.array(X).reshape(len(X),seq_len,1)
    y = np.array(y)
    model_l = Sequential()
    model_l.add(LSTM(32,input_shape=(X.shape[1],X.shape[2])))
    model_l.add(Dense(8,activation="relu"))
    model_l.add(Dense(1))
    model_l.compile(optimizer="adam",loss="mse")
    model_l.fit(X,y,epochs=epochs,batch_size=16,verbose=verbose)
    last_seq = scaled[-seq_len:].reshape(seq_len,1)
    preds_scaled=[]
    cur_seq=last_seq.copy()
    for _ in range(days_ahead):
        x_in = cur_seq.reshape(1,seq_len,1)
        p = model_l.predict(x_in,verbose=0)[0,0]
        preds_scaled.append(p)
        cur_seq = np.vstack([cur_seq[1:],[[p]]])
    if scaler_local:
        inv = scaler_local.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten().tolist()
    else:
        inv = (np.array(preds_scaled)*(series.max()-series.min()+1e-6)+series.min()).tolist()
    return [float(x) for x in inv], None

# ---------------- Recommendations ----------------
def recommendation_for_aqi(aqi):
    """
    Returns a tuple (level_text, short_advice, activity_suggestion)
    Based on Indian AQI Standard:
    0–50 Good
    51–100 Satisfactory
    101–200 Moderate
    201–300 Poor
    301–400 Very Poor
    401+ Severe / Hazardous
    """

    # Missing or invalid AQI
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return (
            "Unknown",
            "AQI data not available.",
            "No recommendation possible."
        )

    try:
        aqi = float(aqi)
    except:
        return (
            "Unknown",
            "AQI data invalid.",
            "No recommendation possible."
        )

    # ---------------- AQI Category Logic ----------------

    if 0 <= aqi <= 50:
        return (
            "Good",
            "Air quality is good. Minimal precautions needed.",
            "Outdoor: All activities ✔ | Indoor: Normal activity"
        )

    if 51 <= aqi <= 100:
        return (
            "Satisfactory",
            "Air quality is acceptable. Minor concern for very sensitive people.",
            "Outdoor: Normal activities ✔ | Indoor: Normal activity"
        )

    if 101 <= aqi <= 200:
        return (
            "Moderate",
            "Air quality may cause discomfort to sensitive groups.",
            "Outdoor: Limit long/intense activities ❌ | Indoor: Keep windows closed if sensitive"
        )

    if 201 <= aqi <= 300:
        return (
            "Poor",
            "Air quality unhealthy for sensitive groups. Reduce exposure.",
            "Outdoor: Avoid long/intense activity ❌ | Indoor: Use AC/filters, keep windows closed"
        )

    if 301 <= aqi <= 400:
        return (
            "Very Poor",
            "Air quality is very unhealthy. Serious health impacts for sensitive groups.",
            "Outdoor: Avoid outdoor activity ❌ | Indoor: Use air purifier if possible"
        )

    # **AQI > 400**
    return (
        "Hazardous",
        "Severe air quality. High risk of health impacts.",
        "Outdoor: Stay indoors ❌ | Indoor: Use high-efficiency filters; N95 if going out"
    )


# ---------------- UI Tabs ----------------
tab1,tab2,tab3,tab4 = st.tabs(["Real-Time Prediction","AQI Trend (History)","Hotspot Map","Seasonal Insights & LSTM Forecast"])

# ---------------- TAB1 Real-Time Prediction ----------------
with tab1:
    st.header("🔎 Real-Time AQI Prediction")

    city_input = st.text_input("Enter city:", value="noida")

    if st.button("Predict AQI Now", key="predict_btn"):

        pollutants, err = fetch_aqi_pollutants(city_input.strip(), AQICN_API_TOKEN)

        if pollutants is None:
            st.error(f"Could not fetch pollutants: {err}")

        else:
            # Show pollutant table
            st.subheader(f"Pollutants for {city_input.title()}")
            st.table(pd.DataFrame([pollutants]).T.rename(columns={0: "Value"}))

            # Prediction
            pred_cat, numeric_aqi, pred_err = predict_category_numeric(pollutants)

            if pred_err:
                st.error(pred_err)
            else:

                # ---------------- Color Map ----------------
                color_map = {
                    "Good": "#2ecc71",
                    "Moderate": "#f1c40f",
                    "Poor": "#e67e22",
                    "Very Poor": "#d35400",
                    "Hazardous": "#e74c3c"
                }

                # Main AQI Box
                display_cat = pred_cat if pred_cat else "Unknown"

                st.markdown(
                    f"""
                    <div style='padding:10px;border-radius:10px;background:{color_map.get(display_cat,"#7f8c8d")};
                    color:white;margin-bottom:15px'>
                        <h3 style='margin:0'>Predicted AQI: {display_cat} ({numeric_aqi})</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ---------------- Pollutant Bar Chart ----------------
                poll_df = pd.DataFrame({
                    "Pollutant": list(pollutants.keys()),
                    "Value": list(pollutants.values())
                }).set_index("Pollutant")

                if PLOTLY_AVAILABLE:
                    fig = px.bar(
                        poll_df.reset_index(),
                        x="Pollutant", y="Value",
                        title="Pollutant Levels", template="plotly_white"
                    )
                    fig.update_traces(marker_color="steelblue")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(poll_df)

                # ---------------- HEALTH & ACTIVITY SUGGESTIONS ----------------
                level, short_advice, activity = recommendation_for_aqi(numeric_aqi)

                # Custom Icon + Message Boxes
                icon_map = {
                    "Good": "🟢",
                    "Moderate": "🟡",
                    "Poor": "🟠",
                    "Very Poor": "🔴",
                    "Hazardous": "🛑"
                }

                st.subheader("Health & Activity Recommendation")

                st.markdown(
                    f"""
                    <div style="padding:10px;border-radius:10px;background:{color_map.get(level, "#7f8c8d")};
                    color:white;margin-top:10px">
                        <h3 style="margin:0">{icon_map.get(level,'ℹ️')} {level} Air Quality</h3>
                        <p style="margin:5px 0;font-size:16px"><b>Advice:</b> {short_advice}</p>
                        <p style="margin:5px 0;font-size:16px"><b>Activity:</b> {activity}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Additional Hazardous Alert
                if level == "Hazardous":
                    st.warning("⚠️ WARNING: Air quality is extremely dangerous. Stay indoors!")

                # ---------------- SAVE REAL-TIME AQI TO CSV ----------------
                hist = load_history()

                try:
                    new_row = {
                        "City": city_input.title(),
                        "Date": pd.Timestamp(datetime.utcnow().date()),
                        "PM2.5": pollutants.get("PM2.5", np.nan),
                        "PM10": pollutants.get("PM10", np.nan),
                        "NO2": pollutants.get("NO2", np.nan),
                        "CO": pollutants.get("CO", np.nan),
                        "O3": pollutants.get("O3", np.nan),
                        "AQI": numeric_aqi
                    }

                    if hist is None or hist.empty:
                        out_df = pd.DataFrame([new_row])
                    else:
                        out_df = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

                    out_df.to_csv(HIST_CSV, index=False)
                    st.success(f"Real-time AQI entry for {city_input.title()} saved.")

                except Exception as e:
                    st.error(f"Could not save data: {e}")



# ---------------- TAB2 AQI Trend ----------------
with tab2:
    st.header("📈 AQI Trend")

    hist = load_history()

    if hist.empty:
        st.info("Historical CSV not available.")
    else:

        # Clean column names (strip spaces)
        hist.columns = hist.columns.str.strip()

        # Convert Date column
        hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")

        # Convert ALL pollutant columns to numeric
        pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
        for col in pollutant_cols:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")

        # Convert AQI numeric
        if "AQI" in hist.columns:
            hist["AQI"] = pd.to_numeric(hist["AQI"], errors="coerce")

        # --- City selection ---
        city_list = sorted(hist["City"].dropna().unique())
        sel_city = st.selectbox("Select City:", city_list, key="tab2_city")

        city_df = hist[hist["City"] == sel_city].sort_values("Date")

        # --- AQI Trend ---
        st.subheader("Overall AQI Trend")
        if "AQI" in city_df.columns:
            fig = px.line(city_df, x="Date", y="AQI",
                          title=f"{sel_city} AQI Trend",
                          template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- POLLUTANT TRENDS ---
        st.subheader("Pollutant Trends")

        pollutants = ["PM2.5", "PM10", "NO2", "CO", "O3"]

        for col in pollutants:
            if col not in city_df.columns:
                continue

            series = city_df[col].dropna()

            if series.empty:
                st.warning(f"No valid data for {col}")
                continue

            fig2 = px.line(city_df, x="Date", y=col,
                           title=f"{col} Trend",
                           template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)



# ---------------- TAB3 Hotspot Map ----------------
# ---------------- TAB3 Hotspot Map + K-Means + FP-Growth ----------------
with tab3:
    st.header("📊 Advanced Data Insights")

    hist = load_history()

    if not hist.empty:

        # ------------------ 🔵 1. Pollution Hotspot Map ------------------
        st.subheader("🗺️ Pollution Hotspots Map (Latest Day)")
        if "Date" in hist.columns:
            latest_date = hist["Date"].max()
            recent = hist[hist["Date"] == latest_date]
            if recent.empty:
                recent = hist.copy()
        else:
            recent = hist.copy()

        hotspot = recent.groupby("City").agg({"AQI":"mean","PM2.5":"mean"}).reset_index()
        hotspot = ensure_latlon(hotspot)
        hotspot["size"] = safe_marker_size(hotspot["PM2.5"])

        if PLOTLY_AVAILABLE:
            fig_map = px.scatter_mapbox(
                hotspot, lat="lat", lon="lon", size="size", color="AQI",
                hover_name="City", zoom=4, height=500,
                title="Pollution Hotspot Map", template="plotly_white"
            )
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.map(hotspot.rename(columns={"lat":"latitude","lon":"longitude"})[["latitude","longitude"]])

        st.markdown("---")

        # ------------------ 🔵 2. K-Means Hotspot Clustering ------------------
        st.subheader("📌 K-Means Pollution Cluster Analysis")

        if "AQI" in hist.columns:
            df_k = hist.copy()
            df_k = df_k.dropna(subset=["AQI"])

            # Select numeric features for clustering
            features = df_k[["AQI"]].values

            inertias = []
            K = range(1, 6)
            for k in K:
                km = KMeans(n_clusters=k, random_state=42)
                km.fit(features)
                inertias.append(km.inertia_)

            # Elbow graph
            st.write("### 📉 Elbow Method (Optimal K)")
            if PLOTLY_AVAILABLE:
                fig_elbow = px.line(
                    x=list(K), y=inertias, markers=True,
                    title="Elbow Curve for Optimal Clusters", template="plotly_white"
                )
                fig_elbow.update_xaxes(title="Number of Clusters (K)")
                fig_elbow.update_yaxes(title="Inertia")
                st.plotly_chart(fig_elbow, use_container_width=True)
            else:
                st.line_chart(pd.DataFrame({"K": list(K), "Inertia": inertias}).set_index("K"))

            # Run KMeans with K=3 (best general choice)
            kmeans = KMeans(n_clusters=3, random_state=42)
            df_k["Cluster"] = kmeans.fit_predict(features)

            st.write("### 📍 Assigned Clusters:")
            st.dataframe(df_k[["City", "Date", "AQI", "Cluster"]].head(20))

            # Save result
            df_k.to_csv("kmeans_clusters.csv", index=False)
            st.success("✅ K-Means clusters saved as kmeans_clusters.csv")

        st.markdown("---")

        # ------------------ 🔵 3. FP-Growth Frequent Pollution Patterns ------------------
        st.subheader("🧠 FP-Growth – Frequent Pollution Patterns")

        try:
            df_fp = hist.copy()

            # Convert to boolean columns
            df_fp["PM2.5_HIGH"] = df_fp["PM2.5"] > df_fp["PM2.5"].mean()
            df_fp["PM10_HIGH"]  = df_fp["PM10"] > df_fp["PM10"].mean()
            df_fp["NO2_HIGH"]   = df_fp["NO2"] > df_fp["NO2"].mean()
            df_fp["SO2_HIGH"]   = df_fp["SO2"] > df_fp["SO2"].mean()
            df_fp["CO_HIGH"]    = df_fp["CO"] > df_fp["CO"].mean()
            df_fp["O3_HIGH"]    = df_fp["O3"] > df_fp["O3"].mean()

            bool_df = df_fp[[
                "PM2.5_HIGH","PM10_HIGH","NO2_HIGH",
                "SO2_HIGH","CO_HIGH","O3_HIGH"
            ]].astype(bool)

            # FP-Growth
            patterns = fpgrowth(bool_df, min_support=0.1, use_colnames=True)
            patterns = patterns.sort_values("support", ascending=False)

            st.write("### 📂 Frequent Itemsets")
            st.dataframe(patterns)

            # Display as beautiful cards
            st.write("### 🔥 Top Pollution Pattern Insights")

            for _, row in patterns.head(8).iterrows():
                st.info(f"**Pattern:** {list(row['itemsets'])} | **Support:** {round(row['support'], 3)}")


            # Save outputs
            patterns.to_csv("fp_growth_patterns.csv", index=False)
            st.success("✅ FP-Growth output saved as fp_growth_patterns.csv")

        except Exception as e:
            st.error(f"FP-Growth error: {e}")

    else:
        st.info("Historical CSV not available.")


# ---------------- TAB4 Seasonal Insights & LSTM ----------------
with tab4:
    st.header("🎯 Seasonal Insights & 7-Day Forecast (LSTM)")
    hist = load_history()
    if not hist.empty:
        # Seasonal Insights
        st.subheader("📊 Month-wise Average AQI (Seasonal Pattern)")
        hist["Month"] = hist["Date"].dt.month
        month_avg = hist.groupby("Month")["AQI"].mean().reindex(range(1,13)).fillna(0)
        month_df = pd.DataFrame({"Month": month_avg.index, "Avg_AQI": month_avg.values})
        if PLOTLY_AVAILABLE:
            fig = px.bar(month_df, x="Month", y="Avg_AQI", title="Average AQI by Month", labels={"Avg_AQI":"AQI"}, template="plotly_white")
            fig.update_traces(marker_color="steelblue")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(month_df.set_index("Month")["Avg_AQI"])

        # Diwali Spike
        st.subheader("🎆 Diwali / November AQI Spike")
        hist["Year"] = hist["Date"].dt.year
        diwali_summary = []
        for y in sorted(hist["Year"].unique()):
            oct_mean = hist[(hist["Year"]==y) & (hist["Month"]==10)]["AQI"].mean()
            nov_mean = hist[(hist["Year"]==y) & (hist["Month"]==11)]["AQI"].mean()
            oct_val = 0.0 if pd.isna(oct_mean) else float(oct_mean)
            nov_val = 0.0 if pd.isna(nov_mean) else float(nov_mean)
            diwali_summary.append({
                "Year": int(y),
                "Oct_Mean_AQI": round(oct_val,2),
                "Nov_Mean_AQI": round(nov_val,2)
            })
        st.table(pd.DataFrame(diwali_summary))

        # LSTM Forecast
        st.subheader("📅 7-Day AQI Forecast (LSTM)")
        city_for_forecast = st.selectbox("Select city for forecast:", sorted(hist["City"].unique()), key="forecast_city_tab4")
        forecast_btn = st.button("Run 7-day LSTM forecast", key="lstm_btn_tab4")
        if forecast_btn:
            city_df = hist[hist["City"]==city_for_forecast].sort_values("Date").reset_index(drop=True)
            if len(city_df)<60:
                st.error(f"Not enough historical records ({len(city_df)}), need ~60 for LSTM")
            else:
                with st.spinner("Training LSTM and forecasting..."):
                    preds, err = lstm_forecast_for_city(city_df, days_ahead=7, epochs=25, verbose=0)
                    if preds is None:
                        st.error(err)
                    else:
                        start_date = city_df["Date"].max() + pd.Timedelta(days=1)
                        dates = [start_date + pd.Timedelta(days=i) for i in range(7)]
                        table = pd.DataFrame({"Date": dates, "Predicted_AQI": np.round(preds,2)})
                        def cat_of(aqi):
                            if aqi<=50: return "Good"
                            if aqi<=100: return "Moderate"
                            if aqi<=200: return "Poor"
                            if aqi<=300: return "Very Poor"
                            return "Hazardous"
                        table["Category"] = table["Predicted_AQI"].apply(cat_of)
                        # add recommendations column
                        recs = [recommendation_for_aqi(v)[1] for v in table["Predicted_AQI"]]
                        table["Advice"] = recs
                        st.write("7-Day Forecast (LSTM):")
                        st.table(table)

                        if PLOTLY_AVAILABLE:
                            hist_recent = city_df.tail(60)[["Date","AQI"]].copy()
                            future_df = pd.DataFrame({"Date": dates, "AQI": table["Predicted_AQI"].values})
                            hist_future = pd.concat([hist_recent, future_df], ignore_index=True)
                            fig = px.line(hist_future, x="Date", y="AQI", title=f"{city_for_forecast} - Recent AQI + Forecast", template="plotly_white")
                            fig.update_traces(line=dict(color="royalblue", width=3))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.line_chart(pd.Series(list(city_df.tail(60)["AQI"])+list(table["Predicted_AQI"])))
                            
    else:
        st.info("Historical CSV not available.")
        
        

st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().isoformat()}Z")
