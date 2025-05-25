# AI-Powered-Intrusion-Detection-System-IDS
This project implements a web-based Intrusion Detection System (IDS) that uses a Convolutional Neural Network (CNN) to detect malicious network traffic. It allows users to upload flow-based network data (CSV files) and returns predictions indicating whether the traffic is benign or an attack.

#### DEMO 


https://github.com/user-attachments/assets/13d99b98-0f52-4154-85fc-d6ef4c271d18

<img width="748" alt="Screenshot 2025-05-26 021001" src="https://github.com/user-attachments/assets/00e54e39-4919-4ccc-acc2-b78b9eabab8b" />

<img width="748" alt="Screenshot 2025-05-26 021122" src="https://github.com/user-attachments/assets/0ac74312-ed11-43af-8283-55e33c1bf3e5" />

<img width="748" alt="Screenshot 2025-05-26 021225" src="https://github.com/user-attachments/assets/1bf2299a-d719-423f-b51e-16dac3101e44" />






## 🚀 Features

- Deep Learning-based classification (CNN)
- Flask web interface
- CSV file upload for offline analysis
- Clean prediction output
- Modular and extensible code structure

## 🧠 Model Details

- **Model Type**: Convolutional Neural Network (CNN)
- **Input Features**:
  - Flow Duration
  - Total Fwd Packets
  - Total Backward Packets
  - Fwd Packet Length Max
  - Bwd Packet Length Max
  - Flow Bytes/s

  ## 📦 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/SamGitH02/AI-powered-Network-Intrusion-Detecion-System.git
   cd AI-powered-Network-Intrusion-Detecion-System
   ```


2. **Setup Instructions**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

```

3. **Install dependencies**
```bash
pip install -r requirements.txt
``` 
4. **Run the application**
```
python app.py
```
5. **Access the web interface**
```
Visit http://127.0.0.1:5000 in your browser
```
6. **Future Enhancements**
- **Real-time packet capture using Scapy**
- **Live traffic conversion via CICFlowMeter**
- **Dashboard and alert system for active monitoring**



