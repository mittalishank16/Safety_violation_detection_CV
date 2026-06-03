from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db, ViolationLog
from inference import engine
import cv2, numpy as np, base64, os, tempfile

app = FastAPI(title='Safety Detection API', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

violation_timers: dict = {}
VIOLATION_THRESHOLD_FRAMES = 90

@app.get('/health')
def health(): 
    return {'status': 'ok', 'model': 'loaded'}

@app.post('/analyze-video')
async def analyze_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    
    # Use standard cross-platform temp directory structure
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        violation_timers.clear()
        logged = 0
        frame_idx = 0
        
        cap = cv2.VideoCapture(tmp_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
                
            inp = engine.preprocess(frame)
            detections = engine.postprocess(
                engine.session.run(None, {engine.input_name: inp}), 
                frame.shape
            )
            
            # This loop must be evaluated INSIDE the frame reading pipeline
            for det in detections:
                if det['is_violation']:
                    key = (0, det['class_name'])
                    if key not in violation_timers: 
                        violation_timers[key] = frame_idx
                    elif frame_idx - violation_timers[key] >= VIOLATION_THRESHOLD_FRAMES:
                        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        
                        b64 = ''
                        if crop.size > 0:
                            _, buf = cv2.imencode('.jpg', crop)
                            b64 = base64.b64encode(buf).decode()
                            
                        db.add(ViolationLog(
                            track_id=0, 
                            violation_type=det['class_name'],
                            duration_seconds=round((frame_idx - violation_timers[key]) / fps, 2),
                            bbox_x=(x1 + x2) / 2, 
                            bbox_y=(y1 + y2) / 2, 
                            screenshot_b64=b64
                        ))
                        db.commit()
                        logged += 1
                        violation_timers[key] = frame_idx
            
            # Increment index for every single frame processed
            frame_idx += 1
            
        cap.release()
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return {'frames_processed': frame_idx, 'violations_logged': logged}

@app.get('/violations')
def get_violations(limit: int = 500, db: Session = Depends(get_db)):
    """Return all logged violations for the dashboard."""
    rows = db.query(ViolationLog).order_by(ViolationLog.timestamp.desc()).limit(limit).all()
    return [{
        'id': r.id, 
        'violation_type': r.violation_type,
        'duration_seconds': r.duration_seconds,
        'timestamp': r.timestamp.isoformat(),
        'bbox_x': r.bbox_x, 
        'bbox_y': r.bbox_y,
    } for r in rows]