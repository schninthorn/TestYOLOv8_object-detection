import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLO
model = YOLO("best.pt")  # ใส่โมเดลที่ train แล้ว เช่น best.pt

# เปิดกล้อง (กล้อง 0 คือกล้องหลัก)
cap = cv2.VideoCapture(0)

# ตรวจสอบว่ากล้องเปิดได้ไหม
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

# อ่านภาพจากกล้องทีละเฟรม
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ไม่สามารถอ่านเฟรมจากกล้องได้")
        break

    # ใช้ YOLO ตรวจจับวัตถุ
    results = model(frame)

    # เริ่มต้นจากเฟรมต้นฉบับ (เพื่อวาดข้อความเพิ่มเติม)
    annotated_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]

            # ดึงพิกัดกล่อง (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            print(f"ตรวจพบ {label} ด้วยความมั่นใจ {confidence:.2f} ที่ตำแหน่ง x={center_x}, y={center_y}")

            # ตัวอย่างเงื่อนไขพิเศษ
            if label == 'LED_light':
                #การแสดงกรอบ 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128,0,128), 2)
                text = f"{label} ({confidence:.2f})"
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,0,128), 2)
                cv2.putText(annotated_frame, f"x={center_x}, y={center_y}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if confidence > 0.80:
                    print("✅ Oooook~~~~ it is LED_light")
                else:
                    print("❌ Naaaaaaa~~~~ it not is LED_light")


            if label == 'motorcycle_tube':
                #การแสดงกรอบ 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128,0,0), 2)
                text = f"{label} ({confidence:.2f})"
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,0,0), 2)
                cv2.putText(annotated_frame, f"x={center_x}, y={center_y}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if confidence > 0.80:
                    print("✅ Oooook~~~~ it is motorcycle_tube")
                else:
                    print("❌ Naaaaaaa~~~~ it not is motorcycle_tube")


            if label == 'multitester':
                #การแสดงกรอบ 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,0,255), 2)
                text = f"{label} ({confidence:.2f})"
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.putText(annotated_frame, f"x={center_x}, y={center_y}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if confidence > 0.80:
                    print("✅ Oooook~~~~ it is multitester")
                else:
                    print("❌ Naaaaaaa~~~~ it not is multitester")


            if label == 'shell_brake':
                #การแสดงกรอบ 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,165,255), 2)
                text = f"{label} ({confidence:.2f})"
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                cv2.putText(annotated_frame, f"x={center_x}, y={center_y}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if confidence > 0.80:
                    print("✅ Oooook~~~~ it is shell_brake")
                else:
                    print("❌ Naaaaaaa~~~~ it not is shell_brake")




    # แสดงผล
    cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    # ถ้ากด q หรือ ESC จะออกจากลูป
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') : #or key == 27:
        break

# ปิดกล้องและหน้าต่างแสดงผล
cap.release()
cv2.destroyAllWindows()
