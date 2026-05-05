# 🚀 AI Scooty & Drone Delivery Simulation

An interactive **Smart Delivery Simulation System** that models real-world logistics using **AI-powered scooty routing, drone delivery, and dynamic traffic conditions**.

Built using **Streamlit + Graph Algorithms + OSMnx**, this project simulates how modern delivery systems can optimize routes under real-world constraints like traffic and weather.

---

## 🌐 Live Demo

👉 [Click here to view the app]([https://your-app-name.streamlit.app](https://koramangala-smart-delivery-3qhkl45dczux4vfkkkxrcv.streamlit.app/))

---

## 🎯 Key Features

### 🚚 Multi-Mode Delivery Simulation
- AI Scooty (traffic-aware routing)
- Petrol Scooty (baseline comparison)
- Drone delivery (direct path routing)

---

### 🧠 Traffic-Aware Routing (Core Feature)
- Real-time congestion simulation
- Dynamic rerouting using graph algorithms
- Traffic levels:
  - 🔴 High (rerouted)
  - 🟠 Medium (rerouted)
  - 🟡 Low (original route)

---

### 🌧️ Weather-Based Drone Decisions
- Drone flight restricted based on:
  - Rainfall 🌧️
  - Wind Speed 🌬️
- Intelligent flight clearance system

---

### 🗺️ Live Map Visualization
- Real-time delivery tracking
- Animated routes using PyDeck
- Multiple deliveries simulated simultaneously

---

### 📊 Data-Driven Dashboard
- Traffic distribution analysis
- Rerouting statistics
- Delivery comparisons

---

### ⚔️ Cost Comparison Engine
- AI Scooty vs Petrol vs Drone
- Identifies most cost-efficient delivery mode

---

## 🧩 Tech Stack

| Category        | Tools Used |
|----------------|-----------|
| Frontend       | Streamlit |
| Data Handling  | Pandas, NumPy |
| Visualization  | PyDeck, Plotly |
| Graph Routing  | NetworkX, OSMnx |
| Geo Processing | GeoPandas, Shapely |
| ML Dependency  | Scikit-learn |

---

## 🏗️ Project Structure
project/
│

├── app.py

├── modules/

│ ├── data_loader.py

│ ├── graph_loader.py

│ ├── weather.py

│ ├── traffic.py

│ ├── simulation.py

│

├── data/

│ └── koramangala_delivery_with_petrol_comparison.csv

│

├── requirements.txt

└── README.md

---

## ⚙️ How It Works

### 1️⃣ Data Loading
- Delivery dataset with traffic & weather variables

### 2️⃣ Graph Construction
- Road network fetched using OpenStreetMap (OSMnx)

### 3️⃣ Traffic Simulation
- Random congestion nodes selected
- Edge weights modified to simulate traffic

### 4️⃣ Route Optimization
- Shortest path computed using NetworkX
- Alternative paths generated under congestion

### 5️⃣ Drone Logic
- Uses straight-line interpolation
- Disabled under unsafe weather

### 6️⃣ Visualization
- Animated routes displayed using PyDeck

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

pip install -r requirements.txt

streamlit run app.py

---

## 🚀 Deployment

Deployed using Streamlit Cloud

---

## 🧠 Key Learning Outcomes

* Graph-based route optimization
* Real-time simulation design
* Handling geospatial data
* Modular Python architecture
* Interactive dashboard development

---

## 👥 Team

Prem Kumar
Araly Akanksha Naidu

---

## 💡 Future Enhancements
Real-time traffic API integration
Machine learning for demand prediction
Route optimization using reinforcement learning
Fleet management system
