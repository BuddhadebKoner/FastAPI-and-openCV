# 📹 FastAPI + OpenCV Exam Monitoring System

A real-time AI-powered exam monitoring system that detects person presence using computer vision. Built with FastAPI backend and React frontend.

## ✨ Key Features

- 🎯 **Real-time Face Detection** - OpenCV-powered person detection
- ⚡ **WebSocket Communication** - Live bidirectional data streaming  
- 🚨 **Smart Alerts** - Automatic warnings for no person/multiple people
- 🎨 **Modern UI** - Clean React interface with live video feed
- 📊 **Alert History** - Comprehensive monitoring logs
- 🔄 **Auto-reconnection** - Robust WebSocket connection handling

## 🛠️ Tech Stack

**Backend:** FastAPI • OpenCV • WebSockets • NumPy  
**Frontend:** React • Vite • Modern CSS  
**Platform:** Windows optimized

## � Quick Start

### 1. Backend Setup
```bash
# Navigate to project
cd d:\CODING\fastapi-opencv

# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py
```
**Backend runs on:** `http://localhost:8000`

### 2. Frontend Setup
```bash
# Navigate to frontend
cd frontend

# Install & start
npm install
npm run dev
```
**Frontend runs on:** `http://localhost:5173`

### 3. Usage
1. Open `http://localhost:5173` in browser
2. Allow camera access when prompted
3. Click **"Start Exam Monitoring"**
4. System monitors for person presence automatically

## 📡 API Reference

| Endpoint | Type | Description |
|----------|------|-------------|
| `/` | GET | API status & info |
| `/health` | GET | Health check |
| `/ws/monitor` | WebSocket | Real-time monitoring |

## 🔍 How It Works

```
📹 Camera Feed → 🎯 Face Detection → 🚨 Alert Analysis → 📊 Real-time Updates
```

1. **Capture**: Frontend streams webcam frames every 2 seconds
2. **Process**: Backend analyzes frames using OpenCV Haar Cascade
3. **Detect**: System identifies 0, 1, or multiple faces
4. **Alert**: Generates warnings based on detection results
5. **Update**: Real-time status updates via WebSocket

## ⚠️ Alert System

| Status | Condition | Action |
|--------|-----------|--------|
| 🔴 **High Alert** | No person detected | Immediate warning |
| 🟡 **Medium Alert** | Multiple people detected | Flag suspicious activity |
| 🟢 **Normal** | Single person detected | Continue monitoring |

## ⚙️ Configuration

### Backend Settings
- **Port**: 8000 (configurable in `main.py`)
- **Detection Model**: Haar Cascade classifier
- **Image Format**: JPEG (80% compression)

### Frontend Settings  
- **WebSocket**: `ws://localhost:8000/ws/monitor`
- **Video Resolution**: 640x480
- **Frame Rate**: 2-second intervals

## � Troubleshooting

**Camera Issues:**
- Grant browser camera permissions
- Close other camera applications
- Refresh browser page

**Connection Issues:**
- Ensure backend is running on port 8000
- Check firewall settings
- Verify WebSocket URL

**Detection Issues:**
- Ensure good lighting
- Position face clearly in frame
- Check OpenCV installation

## 📈 Future Roadmap

- [ ] Eye tracking for attention monitoring
- [ ] Object detection for prohibited items  
- [ ] Voice activity detection
- [ ] Database integration for session logs
- [ ] Multi-user exam rooms
- [ ] Mobile app support
- [ ] Advanced AI models

## � Security Notes

- Environment variables for sensitive config
- CORS properly configured for production
- No sensitive data logged
- Secure WebSocket connections recommended

## 📄 License

Educational use - Feel free to modify and extend.

---

**⚡ Quick Commands:**
- Start Backend: `python main.py` or `start_backend.bat`
- Start Frontend: `cd frontend && npm run dev`
- Install All: `pip install -r requirements.txt && cd frontend && npm install`
