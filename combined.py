from ultralytics import YOLO
import cv2
import time

# LOAD MODELS

ppe_model = YOLO("models/best1.pt")
pose_model = YOLO("yolo11n-pose.pt")

# VIDEO

cap = cv2.VideoCapture("videos/conveyor.MOV")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ex.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)
safe_panel_time = 0
SAFE_MEMORY = 5
danger_time = 0
DANGER_MEMORY = 3

ppe_seen = set()
last_alert_time = {}

ALERT_COOLDOWN = 15
ppe_last_seen = {}

PPE_MEMORY = 30 # seconds
def should_send_alert(alert_name):

    current_time = time.time()

    if alert_name not in last_alert_time:
        last_alert_time[alert_name] = current_time
        return True

    if current_time - last_alert_time[alert_name] > ALERT_COOLDOWN:
        last_alert_time[alert_name] = current_time
        return True

    return False

while True:

    ret, frame = cap.read()

    if not ret:
        break
    ppe_seen.clear()
    detected_classes = []
    electrical_panel_boxes = []
    fencing_boxes = []
    conveyor_boxes = []
    stairs_boxes = []


    # RUN MODELS

    ppe_results = ppe_model(
        frame,
        imgsz=1280,
        conf=0.3
    )
    pose_results = pose_model(frame)

    annotated = ppe_results[0].plot()

    # PPE DETECTIONS

    for box in ppe_results[0].boxes:

        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = ppe_model.names[class_id]
        print(class_name, conf)


        if conf < 0.3:
            continue

        # class_name = ppe_model.names[class_id]
        print("CLASS:", class_name)
        if class_name == "phone":
            print("PHONE DETECTED")

        detected_classes.append(class_name)
        if class_name in [
            "hardhat",
            "gloves",
            "electrical_gloves",
            "jacket",
            "electrical_jacket",
            "safety shoes",
            "goggles",
            "electrical_goggles",
            "mask"
       ]:
             
             ppe_seen.add(class_name)
             ppe_last_seen[class_name] = time.time()
             print("PPE SEEN:", class_name)


        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        if class_name == "electrical panel":

            electrical_panel_boxes.append(
                (x1, y1, x2, y2)
            )

        elif class_name == "fencing":

            fencing_boxes.append(
                (x1, y1, x2, y2)
            )
        elif class_name == "conveyor":
            conveyor_boxes.append(
               (x1, y1, x2, y2)
        
            )
        elif class_name == "stairs":
            stairs_boxes.append(
                (x1, y1, x2, y2)
            )   
    # PPE CHECK

    missing_ppe = []
    print("Detected Classes:", detected_classes)


    # Conveyor Area PPE

    if len(conveyor_boxes) > 0:
        required_ppe = [
        "hardhat",
        "gloves",
        "jacket",
        "safety shoes",
        "goggles"
        ]

# Electrical Area PPE

    elif len(electrical_panel_boxes) > 0:
        required_ppe = [
        "hardhat",
        "electrical_gloves",
        "electrical_jacket",
        "safety shoes",
        "electrical_goggles"
    ]

# Stair Area

    elif len(stairs_boxes) > 0 or "handrail" in detected_classes:
             required_ppe = [
                  "hardhat",
                  "jacket",
                  "safety shoes"
             ]


# General Area

    else:
        required_ppe = [
            "hardhat",
            "jacket",
            "safety shoes"
        ]

    current_time = time.time()

    for item in required_ppe:
        present = False

        if item == "jacket":
            if (
                current_time - ppe_last_seen.get("jacket", 0) < PPE_MEMORY
                or
                current_time - ppe_last_seen.get("electrical_jacket", 0) < PPE_MEMORY         
            ):
                present=True

        elif item == "gloves":
            if (
                current_time - ppe_last_seen.get("gloves", 0) < PPE_MEMORY
                or
                current_time - ppe_last_seen.get("electrical_gloves", 0) < PPE_MEMORY
            ):
                present=True

        elif item == "goggles":
            if (
                current_time - ppe_last_seen.get("goggles", 0) < PPE_MEMORY
                or
                current_time - ppe_last_seen.get("electrical_goggles", 0) < PPE_MEMORY
        ):
                present=True
        else:
            if current_time - ppe_last_seen.get(item, 0) < PPE_MEMORY:
                present = True
        if not present:
            missing_ppe.append(item)

    print("Required PPE:", required_ppe)
    print("Missing PPE:", missing_ppe)
    print("--------------------------------")
    # POSE DETECTION

    if pose_results[0].keypoints is not None:

        keypoints = pose_results[0].keypoints.xy

        for person in keypoints:

            left_wrist = person[9]
            right_wrist = person[10]

            lw_x = int(left_wrist[0])
            lw_y = int(left_wrist[1])

            rw_x = int(right_wrist[0])
            rw_y = int(right_wrist[1])
            # 3 POINT CONTACT CHECK

            if len(stairs_boxes) > 0 or "handrail" in detected_classes:
                hand_on_rail = False
                for box in ppe_results[0].boxes:
                    cls = ppe_model.names[int(box.cls[0])]
                    if cls == "handrail":
                        hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                        margin = 80
                        if (
                            hx1-margin < lw_x < hx2+margin
                            and
                            hy1-margin < lw_y < hy2+margin
                        ):
                            hand_on_rail = True
                        if (
                            hx1-margin < rw_x < hx2+margin
                            and
                            hy1-margin < rw_y < hy2+margin
                        ):
                            hand_on_rail = True

                if not hand_on_rail:
                    if should_send_alert("three_point_contact"):
                        print("ALERT SENT: NO 3 POINT CONTACT")

                cv2.rectangle(
                    annotated,
                    (20,300),
                    (1200,400),
                    (0,0,255),
                    -1
                )
                cv2.putText(
                    annotated,
                    "NO 3 POINT CONTACT",
                    (40,370),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255,255,255),
                    3
                )
            print(
                f"LW=({lw_x},{lw_y})  RW=({rw_x},{rw_y})"
)

            # DRAW WRISTS

            cv2.circle(
                annotated,
                (lw_x, lw_y),
                10,
                (0,255,0),
                -1
            )

            cv2.circle(
                annotated,
                (rw_x, rw_y),
                10,
                (0,0,255),
                -1
            )

            cv2.putText(
                annotated,
                "LEFT WRIST",
                (lw_x, lw_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            cv2.putText(
                annotated,
                "RIGHT WRIST",
                (rw_x, rw_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,255),
                2
            )

            # ELECTRICAL PANEL CHECK

            for panel in electrical_panel_boxes:

                ex1, ey1, ex2, ey2 = panel
                print(
                    f"PANEL=({ex1},{ey1},{ex2},{ey2})"
                )
                
                margin = 80
                if (ex1 - margin < rw_x < ex2 + margin
                   and
                   ey1 - margin < rw_y < ey2 + margin
                ) or (
                    ex1 - margin < lw_x < ex2 + margin
                    and
                    ey1 - margin < lw_y < ey2 + margin
                ):

                    current_time = time.time()

                    electrical_ppe_ok = (
                         "electrical_gloves" in ppe_seen
                         and
                         "electrical_jacket" in ppe_seen
                         and
                         "hardhat" in ppe_seen
                         and
                         "safety shoes" in ppe_seen
                         and
                         "electrical_goggles" in ppe_seen
                    )

                    if electrical_ppe_ok:
                        cv2.rectangle(
                            annotated,
                            (20,20),
                            (1100,140),
                            (0,255,0),
                            -1
                        )
                        cv2.putText(
                            annotated,
                            "SAFE ELECTRICAL WORK",
                            (40,90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255,255,255),
                            5
                        )

                    else:
                        if should_send_alert("electrical_ppe"):
                            print("ALERT SENT: ELECTRICAL PPE REQUIRED")
                        cv2.rectangle(
                            annotated,
                            (20,20),
                            (1100,140),
                            (0,0,255),
                            -1
                       )

                        cv2.putText(
                            annotated,
                            "CRITICAL: ELECTRICAL PPE REQUIRED",
                            (40,90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255,255,255),
                            5
                       )
                        
            if len(stairs_boxes) > 0 or "handrail" in detected_classes:
                if "weight" in detected_classes:
                    cv2.rectangle(
                        annotated,
                        (20,150),
                        (1200,250),
                        (0,0,255),
                        -1
                    )

                    cv2.putText(
                        annotated,
                        "WARNING: LOAD DETECTED ON Hands",
                        (40,220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255,255,255),
                        3
                        )
            # STAIR PHONE CHECK

            if len(stairs_boxes) > 0 or "handrail" in detected_classes:
                if "phone" in detected_classes:
                    cv2.rectangle(
                        annotated,
                        (20,20),
                        (1200,140),
                        (0,0,255),
                        -1
                    )
                    cv2.putText(
                        annotated,
                        "DANGER: PHONE USAGE ON STAIRS",
                        (40,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255,255,255),
                        4
                    )

               
            # FENCING CHECK

            for fence in fencing_boxes:
                fx1, fy1, fx2, fy2 = fence

    # danger boundary inside fence
                danger_line_x = fx1 + int((fx2 - fx1) * 0.55)

    # draw line
                cv2.line(
                    annotated,
                    (danger_line_x, fy1),
                    (danger_line_x, fy2),
                    (0, 0, 255),
                    5
                )
                current_time = time.time()
                if rw_x < danger_line_x or lw_x < danger_line_x:
                    danger_time = time.time()
                if current_time - danger_time < DANGER_MEMORY:
                    if should_send_alert("conveyor_intrusion"):
                        print("ALERT SENT: HAND INSIDE CONVEYOR")
                    cv2.rectangle(
                        annotated,
                        (20,20),
                        (1200,140),
                        (0,0,255),
                        -1
                    )
                    cv2.putText(
                        annotated,
                        "HIGH RISK: HAND INSIDE CONVEYOR ZONE",
                        (40,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (255,255,255),
                        4
                    )
                elif rw_x < danger_line_x + 80 or lw_x < danger_line_x + 80:
                    cv2.rectangle(
                        annotated,
                        (20,20),
                        (1000,140),
                        (0,165,255),
                        -1
                    )
                    cv2.putText(
                        annotated,
                        "WARNING: DO NOT TOUCH FENCE",
                        (40,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (255,255,255),
                        4
                    )
                     
                   
    y_pos = 250
    for item in missing_ppe:
        if should_send_alert(item):
            print("ALERT SENT:", item)

        # later:
        # send_safety_alert(f"MISSING: {item}")

        cv2.putText(
            annotated,
            f"MISSING: {item}",
            (40, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

    y_pos += 50

    for item in missing_ppe:
        cv2.putText(
            annotated,
            f"MISSING: {item}",
            (40, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

        y_pos += 50
    
   

    # DISPLAY
    out.write(annotated)
    cv2.imshow(
        "Industrial Safety Monitoring",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Output video saved as output_demo.mp4")