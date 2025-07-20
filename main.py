from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import json
import asyncio
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Finger Detection System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Finger detection using contour analysis
def detect_fingers(image_data):
    """
    Detect and count fingers using contour analysis and convex hull
    """
    try:
        # Decode base64 image
        img_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Could not decode image"}
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "finger_count": 0,
                "hand_detected": False,
                "confidence": 0
            }
        
        # Find the largest contour (assumed to be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Check if the contour is large enough to be a hand
        if area < 3000:
            return {
                "finger_count": 0,
                "hand_detected": False,
                "confidence": 0
            }
        
        # Calculate convex hull and defects
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)
        
        finger_count = 0
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(largest_contour[s][0])
                end = tuple(largest_contour[e][0])
                far = tuple(largest_contour[f][0])
                
                # Calculate angle between start, far and end points
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                
                # Calculate angle using cosine rule
                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                angle_deg = np.degrees(angle)
                
                # If angle is less than 90 degrees and distance is significant, count as finger
                if angle_deg <= 90 and d > 4000:
                    finger_count += 1
        
        # Add 1 to account for thumb (which usually doesn't create a defect)
        if finger_count > 0:
            finger_count += 1
        
        # Limit finger count to reasonable range (0-5)
        finger_count = min(finger_count, 5)
        
        # Calculate confidence based on area and contour quality
        confidence = min(100, (area / 10000) * 100)
        
        return {
            "finger_count": finger_count,
            "hand_detected": True,
            "confidence": round(confidence, 1),
            "hand_area": area
        }
        
    except Exception as e:
        logger.error(f"Error in finger detection: {str(e)}")
        return {
            "finger_count": 0,
            "hand_detected": False,
            "confidence": 0,
            "error": str(e)
        }

def process_finger_detection(image_data):
    """
    Process finger detection and return result
    """
    try:
        result = detect_fingers(image_data)
        
        # Determine status based on finger detection
        if "error" in result:
            status = "ERROR"
            message = "Camera processing error"
            alert_level = "high"
        elif not result["hand_detected"]:
            status = "NO_HAND"
            message = "👋 Show your hand to the camera"
            alert_level = "low"
        else:
            finger_count = result["finger_count"]
            status = "DETECTED"
            
            if finger_count == 0:
                message = "✊ Fist detected - 0 fingers"
            elif finger_count == 1:
                message = "☝️ One finger detected"
            elif finger_count == 2:
                message = "✌️ Two fingers detected"
            elif finger_count == 3:
                message = "🤟 Three fingers detected"
            elif finger_count == 4:
                message = "🖖 Four fingers detected"
            elif finger_count == 5:
                message = "🖐️ Five fingers detected"
            else:
                message = f"🤚 {finger_count} fingers detected"
            
            alert_level = "none"
        
        return {
            "status": status,
            "message": message,
            "finger_count": result.get("finger_count", 0),
            "hand_detected": result.get("hand_detected", False),
            "confidence": result.get("confidence", 0),
            "alert_level": alert_level,
            "hand_area": result.get("hand_area", 0),
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logger.error(f"Error in finger detection processing: {str(e)}")
        return {
            "status": "ERROR",
            "message": "System error occurred",
            "finger_count": 0,
            "hand_detected": False,
            "confidence": 0,
            "alert_level": "high",
            "hand_area": 0,
            "timestamp": asyncio.get_event_loop().time()
        }

@app.get("/")
async def read_root():
    return {"message": "Finger Detection System API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "finger-detection"}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive image data from client
            data = await websocket.receive_text()
            
            try:
                # Parse JSON data
                message = json.loads(data)
                
                if message.get("type") == "video_frame":
                    # Process the video frame
                    image_data = message.get("data")
                    
                    if image_data:
                        # Analyze the frame
                        analysis_result = process_finger_detection(image_data)
                        
                        # Send analysis result back to client
                        response = {
                            "type": "analysis_result",
                            "data": analysis_result
                        }
                        
                        await manager.send_personal_message(
                            json.dumps(response), 
                            websocket
                        )
                        
                        # Log detections
                        if analysis_result["status"] == "DETECTED":
                            logger.info(f"Fingers detected: {analysis_result['finger_count']}")
                
                elif message.get("type") == "ping":
                    # Respond to ping with pong
                    pong_response = {
                        "type": "pong",
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await manager.send_personal_message(
                        json.dumps(pong_response), 
                        websocket
                    )
                    
            except json.JSONDecodeError:
                error_response = {
                    "type": "error",
                    "message": "Invalid JSON format"
                }
                await manager.send_personal_message(
                    json.dumps(error_response), 
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from monitoring")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
