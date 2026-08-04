import cv2

class Visualizer:
    @staticmethod
    def draw_active_frame(frame, roi_polygon, detections, person_in_roi):
        processed_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            if det["in_roi"]:
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    processed_frame,
                    "PERSON IN FORBIDDEN AREA",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        cv2.polylines(processed_frame, [roi_polygon], True, (255, 255, 0), 2)

        cv2.putText(
            processed_frame,
            "Detection: ACTIVE",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            2,
        )

        if person_in_roi:
            text = "ALERT: PERSON DETECTED IN FORBIDDEN AREA"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.1  # Font boyutu
            thickness = 3     # Çizgi kalınlığı
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            frame_width = processed_frame.shape[1]
            box_x = (frame_width - text_width) // 2
            box_y = 90  # Sol üstteki yazı ile çakışmaması için aşağı kaydırıldı (Önceki: 50)
            
            padding = 18      
            cv2.rectangle(
                processed_frame,
                (box_x - padding, box_y - text_height - padding),
                (box_x + text_width + padding, box_y + baseline + padding),
                (0, 0, 255),
                -1
            )
            
            cv2.putText(
                processed_frame,
                text,
                (box_x, box_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        return processed_frame

    @staticmethod
    def draw_inactive_frame(frame, roi_polygon):
        processed_frame = frame.copy()

        cv2.polylines(
            processed_frame,
            [roi_polygon],
            True,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            processed_frame,
            "Detection: INACTIVE",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            2,
        )

        return processed_frame