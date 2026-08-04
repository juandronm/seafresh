import argparse
from threading import Thread, Lock
import cv2
from forbidden_area_detection import ForbiddenAreaDetector
from visualizer import Visualizer

class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.lock = Lock()
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self.stop()
                break
            
            ret, frame = self.stream.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            
            if not ret:
                self.stop()
                break

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    parser = argparse.ArgumentParser(description="Forbidden Area Detection System")
    parser.add_argument(
        "--rtsp", 
        type=str, 
        default="rtsp://your5eyt:rfg34hg-6he@77.44.64.69:554/cam/realmonitor?channel=7&subtype=0", 
        help="RTSP stream URL or video path"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/roi_coordinates.json", 
        help="Path to the ROI JSON configuration file"
    )
    
    args = parser.parse_args()

    detector = ForbiddenAreaDetector(rtsp_url=args.rtsp, config_path=args.config)
    
    if not detector.rtsp_url:
        print("Hata: RTSP adresi belirtilmedi!")
        return

    vs = VideoStream(detector.rtsp_url).start()

    print(f"RTSP connection established: {detector.rtsp_url}")
    print(f"Loaded ROI config: {detector.config_path}")

    try:
        while True:
            ret, frame = vs.read()

            if not ret or frame is None:
                print("Frame could not be read or stream ended.")
                break

            if detector.is_detection_time():
                detections, person_in_roi = detector.detect(frame)
                processed_frame = Visualizer.draw_active_frame(
                    frame, detector.roi_polygon, detections, person_in_roi
                )
            else:
                processed_frame = Visualizer.draw_inactive_frame(
                    frame, detector.roi_polygon
                )

            cv2.imshow("Forbidden Area Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Program terminated.")
                break

    finally:
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()