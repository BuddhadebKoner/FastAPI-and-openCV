import React, { useState, useRef, useEffect } from 'react';
import './App.css';

const App = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [detectionData, setDetectionData] = useState(null);
  const [fingerHistory, setFingerHistory] = useState([]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // WebSocket connection
  const connectWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8000/ws/monitor');

      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        console.log('Connected to finger detection server');
      };

      wsRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'analysis_result') {
          setDetectionData(message.data);

          // Add to finger history for tracking
          if (message.data.hand_detected) {
            const newEntry = {
              id: Date.now(),
              count: message.data.finger_count,
              timestamp: new Date().toLocaleTimeString(),
              confidence: message.data.confidence
            };
            setFingerHistory(prev => [newEntry, ...prev.slice(0, 9)]); // Keep last 10 detections
          }
        }
      };

      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        console.log('Disconnected from finger detection server');
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };
    } catch (error) {
      console.error('Failed to connect:', error);
      setConnectionStatus('error');
    }
  };

  // Start video stream
  const startVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Could not access camera. Please ensure camera permissions are granted.');
    }
  };

  // Stop video stream
  const stopVideo = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  // Capture and send frame
  const captureAndSend = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    // Send to server
    if (wsRef.current.readyState === WebSocket.OPEN) {
      const message = {
        type: 'video_frame',
        data: imageData
      };
      wsRef.current.send(JSON.stringify(message));
    }
  };

  // Start finger detection
  const startDetection = async () => {
    setIsDetecting(true);
    connectWebSocket();
    await startVideo();

    // Start sending frames every 1 second for better finger detection
    intervalRef.current = setInterval(captureAndSend, 1000);
  };

  // Stop finger detection
  const stopDetection = () => {
    setIsDetecting(false);

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
    }

    stopVideo();
    setDetectionData(null);
    setFingerHistory([]);
  };

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'DETECTED': return '#4CAF50';
      case 'NO_HAND': return '#FF9800';
      case 'ERROR': return '#F44336';
      default: return '#757575';
    }
  };

  const getFingerEmoji = (count) => {
    switch (count) {
      case 0: return '✊';
      case 1: return '☝️';
      case 2: return '✌️';
      case 3: return '🤟';
      case 4: return '🖖';
      case 5: return '🖐️';
      default: return '🤚';
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>👋 Finger Detection System</h1>
        <div className="status-indicators">
          <div className={`status-indicator ${connectionStatus}`}>
            <span className="status-dot"></span>
            Server: {connectionStatus}
          </div>
          {isDetecting && (
            <div className="detection-mode-indicator">
              <span className="detection-dot"></span>
              DETECTION ACTIVE
            </div>
          )}
        </div>
      </header>

      <main className="main-content">
        <div className="video-section">
          <div className="video-container">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="video-feed"
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {detectionData && (
              <div className="overlay-info">
                <div
                  className="status-badge"
                  style={{ backgroundColor: getStatusColor(detectionData.status) }}
                >
                  {detectionData.status}
                </div>
                {detectionData.hand_detected && (
                  <div className="finger-count">
                    {getFingerEmoji(detectionData.finger_count)} {detectionData.finger_count}
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="controls">
            {!isDetecting ? (
              <button
                onClick={startDetection}
                className="btn btn-primary"
                disabled={connectionStatus === 'error'}
              >
                🚀 Start Finger Detection
              </button>
            ) : (
              <button
                onClick={stopDetection}
                className="btn btn-danger"
              >
                ⏹️ Stop Detection
              </button>
            )}
          </div>
        </div>

        <div className="detection-panel">
          <div className="info-section">
            <h3>📊 Detection Status</h3>
            {detectionData ? (
              <div className="status-info">
                <p><strong>Status:</strong>
                  <span style={{ color: getStatusColor(detectionData.status) }}>
                    {detectionData.status}
                  </span>
                </p>
                <p><strong>Message:</strong> {detectionData.message}</p>
                {detectionData.hand_detected && (
                  <>
                    <p><strong>Fingers Detected:</strong>
                      <span className="finger-display">
                        {getFingerEmoji(detectionData.finger_count)} {detectionData.finger_count}
                      </span>
                    </p>
                    <p><strong>Confidence:</strong> {detectionData.confidence}%</p>
                    <p><strong>Hand Area:</strong> {detectionData.hand_area} pixels</p>
                  </>
                )}
              </div>
            ) : (
              <p className="no-data">No detection data available</p>
            )}
          </div>

          <div className="history-section">
            <h3>📝 Detection History</h3>
            <div className="history-list">
              {fingerHistory.length > 0 ? (
                fingerHistory.map(entry => (
                  <div
                    key={entry.id}
                    className="history-item"
                  >
                    <div className="history-finger">
                      {getFingerEmoji(entry.count)} {entry.count} fingers
                    </div>
                    <div className="history-details">
                      <span>Confidence: {entry.confidence}%</span>
                      <span className="history-time">{entry.timestamp}</span>
                    </div>
                  </div>
                ))
              ) : (
                <p className="no-history">No detection history</p>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Real-time finger counting with computer vision</p>
      </footer>
    </div>
  );
};

export default App;
