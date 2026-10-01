from ultralytics import YOLO
import cv2
import time
import numpy as np
import csv
import os
from datetime import datetime

# LOAD MODELS
ppe_model = YOLO("models/best.pt")
pose_model = YOLO("yolo11n-pose.pt")

# VIDEO SETUP
cap = cv2.VideoCapture("warn_con1.MOV")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "warn_con1.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# CSV LOGGING SETUP
CSV_FILE = "safety_violations.csv"
csv_log_count = 0           # Counter to track how many alerts we've saved
MAX_CSV_ALERTS = 3          
csv_last_logged_time = {}   # Tracks when an alert was last written
CSV_LOG_COOLDOWN = 60       # Wait 60 seconds between identical alerts

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Alert_Type", "Alert_Message"])

def log_to_csv(alert_type, message):
    global csv_log_count, csv_last_logged_time
    
    if csv_log_count >= MAX_CSV_ALERTS:
        return
        
    alert_id = f"{alert_type}_{message}"
    current_time = time.time()
    
    if current_time - csv_last_logged_time.get(alert_id, 0) < CSV_LOG_COOLDOWN:
        return
    
    # LOG THE ALERT: Increment counter and save to file
    csv_log_count += 1
    csv_last_logged_time[alert_id] = current_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, alert_type, message])

init_csv()

# TIMING AND MEMORY VARIABLES
ppe_seen = set()
ppe_last_seen = {}
PPE_MEMORY = 120

# Conveyor movement detection
prev_frame_gray = None
conveyor_moving = False
CONVEYOR_MOVEMENT_THRESHOLD = 1.8 
MOVEMENT_HISTORY_SIZE = 10
movement_history = []

# STICKY ELECTRICAL ZONE TIMER
last_electrical_seen = 0
ELECTRICAL_ZONE_STICKY_TIME = 5.0 

# HELPER FUNCTIONS

def point_in_box(px, py, box, margin=0):
    x1, y1, x2, y2 = box
    return (x1 - margin <= px <= x2 + margin and y1 - margin <= py <= y2 + margin)

def detect_conveyor_movement(frame, conveyor_boxes):
    global prev_frame_gray, movement_history
    if not conveyor_boxes: movement_history.clear(); return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_frame_gray is None: prev_frame_gray = gray.copy(); return False
    total_movement, valid_boxes = 0, 0
    for x1, y1, x2, y2 in conveyor_boxes:
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(int(gray.shape[1]), int(x2)), min(int(gray.shape[0]), int(y2))
        if x2 <= x1 or y2 <= y1: continue
        total_movement += np.mean(cv2.absdiff(prev_frame_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2]))
        valid_boxes += 1
    prev_frame_gray = gray.copy()
    if not valid_boxes: return False
    movement_history.append(total_movement / valid_boxes)
    if len(movement_history) > MOVEMENT_HISTORY_SIZE: movement_history.pop(0)
    return np.mean(movement_history) > CONVEYOR_MOVEMENT_THRESHOLD if len(movement_history) >= 3 else False

def check_three_point_contact_foolproof(person_keypoints, handrail_boxes_raw=None, stairs_box=None):
    l_wrist, r_wrist = person_keypoints[9], person_keypoints[10]
    l_ankle, r_ankle = person_keypoints[15], person_keypoints[16]
    l_shoulder, r_shoulder = person_keypoints[5], person_keypoints[6]
    l_hip, r_hip = person_keypoints[11], person_keypoints[12]
    l_elbow, r_elbow = person_keypoints[7], person_keypoints[8]
    
    def is_valid(kp): return float(kp[0]) > 0 and float(kp[1]) > 0
    contact_points, details = 0, []
    
    if stairs_box:
        if is_valid(l_ankle) and point_in_box(float(l_ankle[0]), float(l_ankle[1]), stairs_box, 150): contact_points += 1; details.append("L.FOOT")
        if is_valid(r_ankle) and point_in_box(float(r_ankle[0]), float(r_ankle[1]), stairs_box, 150): contact_points += 1; details.append("R.FOOT")
    else:
        if is_valid(l_ankle): contact_points += 1; details.append("L.FOOT")
        if is_valid(r_ankle): contact_points += 1; details.append("R.FOOT")

    hand_contact = False
    if is_valid(l_shoulder) and is_valid(r_shoulder) and is_valid(l_hip) and is_valid(r_hip):
        torso_center_x = (float(l_shoulder[0]) + float(r_shoulder[0]) + float(l_hip[0]) + float(r_hip[0])) / 4
        shoulder_width = abs(float(l_shoulder[0]) - float(r_shoulder[0]))
        reach_threshold = shoulder_width * 1.2 
        if is_valid(l_wrist) and is_valid(l_elbow):
            if float(l_shoulder[0]) > float(l_elbow[0]) > float(l_wrist[0]):
                if abs(float(l_wrist[0]) - torso_center_x) > reach_threshold: contact_points += 1; details.append("L.HAND"); hand_contact = True
        if is_valid(r_wrist) and is_valid(r_elbow) and not hand_contact:
            if float(r_shoulder[0]) < float(r_elbow[0]) < float(r_wrist[0]):
                if abs(float(r_wrist[0]) - torso_center_x) > reach_threshold: contact_points += 1; details.append("R.HAND"); hand_contact = True
    if not hand_contact and handrail_boxes_raw:
        for rail in handrail_boxes_raw:
            if is_valid(l_wrist) and point_in_box(float(l_wrist[0]), float(l_wrist[1]), rail, 120): contact_points += 1; details.append("L.HAND-RAIL"); hand_contact = True; break
            if is_valid(r_wrist) and point_in_box(float(r_wrist[0]), float(r_wrist[1]), rail, 120): contact_points += 1; details.append("R.HAND-RAIL"); hand_contact = True; break
    return contact_points >= 3, contact_points, details

def check_phone_usage(person_keypoints, phone_boxes):
    l_wrist, r_wrist = person_keypoints[9], person_keypoints[10]
    l_ear, r_ear = person_keypoints[3], person_keypoints[4]
    def is_valid(kp): return float(kp[0]) > 0 and float(kp[1]) > 0
    def dist(p1, p2): return np.sqrt((float(p1[0])-float(p2[0]))**2 + (float(p1[1])-float(p2[1]))**2)
    shoulder_width = abs(float(person_keypoints[5][0]) - float(person_keypoints[6][0])) if is_valid(person_keypoints[5]) and is_valid(person_keypoints[6]) else 100
    for px1, py1, px2, py2 in phone_boxes:
        cx, cy = (px1+px2)/2, (py1+py2)/2
        if is_valid(l_wrist) and dist(l_wrist, (cx,cy)) < shoulder_width*1.5: return True
        if is_valid(r_wrist) and dist(r_wrist, (cx,cy)) < shoulder_width*1.5: return True
    if is_valid(l_ear) and is_valid(l_wrist) and dist(l_wrist, l_ear) < shoulder_width*0.6: return True
    if is_valid(r_ear) and is_valid(r_wrist) and dist(r_wrist, r_ear) < shoulder_width*0.6: return True
    return False

def check_carrying_weight(person_keypoints, weight_boxes):
    l_wrist, r_wrist = person_keypoints[9], person_keypoints[10]
    def is_valid(kp): return float(kp[0]) > 0 and float(kp[1]) > 0
    def dist(p1, p2): return np.sqrt((float(p1[0])-float(p2[0]))**2 + (float(p1[1])-float(p2[1]))**2)
    shoulder_width = abs(float(person_keypoints[5][0]) - float(person_keypoints[6][0])) if is_valid(person_keypoints[5]) and is_valid(person_keypoints[6]) else 100
    l_occ = any(is_valid(l_wrist) and dist(l_wrist, ((w[0]+w[2])/2, (w[1]+w[3])/2)) < shoulder_width*1.8 for w in weight_boxes)
    r_occ = any(is_valid(r_wrist) and dist(r_wrist, ((w[0]+w[2])/2, (w[1]+w[3])/2)) < shoulder_width*1.8 for w in weight_boxes)
    return {"carrying": l_occ or r_occ, "both_hands": l_occ and r_occ, "one_hand": (l_occ or r_occ) and not (l_occ and r_occ)}

def check_ppe_present(ppe_item, ct, pls, pm):
    if ppe_item == "jacket": return ct - pls.get("jacket", 0) < pm or ct - pls.get("electrical_jacket", 0) < pm
    elif ppe_item == "gloves": return ct - pls.get("gloves", 0) < pm or ct - pls.get("electrical_gloves", 0) < pm
    elif ppe_item == "goggles": return ct - pls.get("goggles", 0) < pm
    else: return ct - pls.get(ppe_item, 0) < pm

def get_missing_ppe(required_items, ct, ppe_last_seen, ppe_memory):
    missing = []
    for item in required_items:
        if not check_ppe_present(item, ct, ppe_last_seen, ppe_memory):
            missing.append(item.upper())
    return missing

# UI DRAWING FUNCTIONS
def draw_bold_alert(frame, text, bg_color, y_pos, x_start=15, fs=1.2, th=3):
    x_start, y_pos = int(x_start), int(y_pos)
    max_w = frame.shape[1] - x_start - 50
    (tw, th2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    if tw > max_w:
        while tw > max_w and len(text) > 10: text = text[:-1]; (tw, th2), _ = cv2.getTextSize(text+"...", cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        text += "..."
    p = 15; bx2 = int(x_start + tw + p*2); by2 = int(y_pos + th2 + p*2)
    if by2 > frame.shape[0] - 20: return y_pos
    cv2.rectangle(frame, (x_start, y_pos), (bx2, by2), bg_color, -1)
    cv2.rectangle(frame, (x_start, y_pos), (bx2, by2), (255,255,255), 4)
    cv2.rectangle(frame, (x_start+4, y_pos+4), (bx2-4, by2-4), tuple(max(0, c-50) for c in bg_color), 2)
    tx, ty = x_start + p, y_pos + th2 + p
    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]: cv2.putText(frame, text, (int(tx+dx), int(ty+dy)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), th+1)
    cv2.putText(frame, text, (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th)
    return by2 + 8

def draw_crit(frame, t, y): return draw_bold_alert(frame, t, (0,0,220), y, fs=1.3, th=4)
def draw_warn(frame, t, y): return draw_bold_alert(frame, t, (0,140,255), y, fs=1.1, th=3)
def draw_safe(frame, t, y): return draw_bold_alert(frame, t, (0,180,0), y, fs=1.0, th=3)

def draw_zone_ind(frame, text, color, x, y, w=250, h=40):
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 3)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    tx, ty = x + (w - tw) // 2, y + 30
    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]: cv2.putText(frame, text, (tx+dx, ty+dy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)

def draw_ppe_panel(frame, ct, pls, pm, zone="NORMAL"):
    px, py, pw = int(frame.shape[1]-380), 15, 365
    
    if zone == "ELECTRICAL":
        check_map = {
            "HARDHAT": "hardhat",
            "ELEC. GLOVES": "electrical_gloves", 
            "ELEC. JACKET": "electrical_jacket",
            "GOGGLES": "goggles",
            "SAFETY SHOES": "safety shoes"
        }
        header_text = "ELEC. ZONE PPE STATUS"
        header_color = (0, 140, 255)
    else:
        check_map = {
            "HARDHAT": "hardhat",
            "JACKET": "jacket",
            "GLOVES": "gloves",
            "GOGGLES": "goggles",
            "SAFETY SHOES": "safety shoes",
            "MASK": "mask"
        }
        header_text = "PPE STATUS"
        header_color = (0, 100, 200)
    
    ph = 50 + (len(check_map) * 38)

    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (30,30,30), -1)
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255,255,255), 4)
    cv2.rectangle(frame, (px, py), (px+pw, py+50), header_color, -1)
    cv2.putText(frame, header_text, (px + 15, py+40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)
    
    iy = py + 85
    for display_name, yolo_class in check_map.items():
        ok = ct - pls.get(yolo_class, 0) < pm
        cv2.rectangle(frame, (px+10, iy-25), (px+pw-10, iy+10), (0,80,0) if ok else (80,0,0), -1)
        cv2.putText(frame, "[OK]" if ok else "[!!]", (px+20, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if ok else (0,0,255), 3)
        cv2.putText(frame, display_name, (px+120, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
        iy += 38

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret: break
    ppe_seen.clear()
    
    crit, warn, safe = [], [], []
    
    ppe_res = ppe_model(frame, imgsz=1280, conf=0.15)
    pose_res = pose_model(frame)
    annotated = ppe_res[0].plot()
    
    el_pan, fence, conv, stairs, rails_raw, phones, weights = [], [], [], [], [], [], []
    glove_boxes_raw, head_ppe_boxes = [], []
    
    for box in ppe_res[0].boxes:
        cid, conf = int(box.cls[0]), float(box.conf[0])
        cname = ppe_model.names[cid]
        if conf < 0.15: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        
        if cname in ["jacket", "electrical_jacket", "safety shoes"]:
            ppe_seen.add(cname); ppe_last_seen[cname] = time.time()
        elif cname in ["gloves", "electrical_gloves"]:
            glove_boxes_raw.append((x1, y1, x2, y2, cname))
        elif cname in ["hardhat", "goggles", "mask"]:
            head_ppe_boxes.append((x1, y1, x2, y2, cname))
            
        if cname in ["electrical panel", "electrical door"]: el_pan.append((x1,y1,x2,y2))
        elif cname == "fencing": fence.append((x1,y1,x2,y2))
        elif cname == "conveyor": conv.append((x1,y1,x2,y2))
        elif cname in ["stairs","staircase","stair"]: stairs.append((x1,y1,x2,y2))
        elif cname.lower() in ["handrail","railing","banister","hand-rail","rail"]: rails_raw.append((x1,y1,x2,y2))
        elif cname in ["phone","mobile","cellphone"]: phones.append((x1,y1,x2,y2))
        elif cname in ["weight","box","package","load","carton"]: weights.append((x1,y1,x2,y2))

    if not stairs and rails_raw:
        stairs.append((min(h[0] for h in rails_raw)-150, min(h[1] for h in rails_raw)-200,
                       max(h[2] for h in rails_raw)+150, max(h[3] for h in rails_raw)+200))

    conv_mov = detect_conveyor_movement(frame, conv)
    for c in conv:
        draw_zone_ind(annotated, "CONVEYOR: MOVING" if conv_mov else "CONVEYOR: STOPPED", (0,150,0) if conv_mov else (0,100,180), c[0], c[1]-50, 320, 45)
    
    ct = time.time()
    in_conv = in_stairs = False
    
    if pose_res[0].keypoints is not None:
        for p in pose_res[0].keypoints.xy:
            lwx, lwy, rwx, rwy = int(p[9][0]), int(p[9][1]), int(p[10][0]), int(p[10][1])
            nose_x, nose_y = int(p[0][0]), int(p[0][1])
            
            cv2.circle(annotated, (lwx,lwy), 12, (0,255,0), -1); cv2.circle(annotated, (lwx,lwy), 15, (255,255,255), 3)
            cv2.circle(annotated, (rwx,rwy), 12, (0,0,255), -1); cv2.circle(annotated, (rwx,rwy), 15, (255,255,255), 3)
            
            def iv(kp): return float(kp[0])>0 and float(kp[1])>0
            
            avg_shoulder_y = None
            if iv(p[5]) and iv(p[6]): avg_shoulder_y = (float(p[5][1]) + float(p[6][1])) / 2
            if iv(p[0]):
                for hx1, hy1, hx2, hy2, hname in head_ppe_boxes:
                    is_worn = False
                    if avg_shoulder_y is not None:
                        if hy2 < avg_shoulder_y + 30: is_worn = True
                    else: is_worn = True
                    if is_worn and point_in_box(nose_x, nose_y, (hx1, hy1, hx2, hy2), margin=40):
                        ppe_seen.add(hname); ppe_last_seen[hname] = time.time(); break 

            for gx1, gy1, gx2, gy2, gname in glove_boxes_raw:
                if point_in_box(lwx, lwy, (gx1, gy1, gx2, gy2), margin=20) or point_in_box(rwx, rwy, (gx1, gy1, gx2, gy2), margin=20):
                    ppe_seen.add(gname); ppe_last_seen[gname] = time.time(); break

            cur_stairs = stairs[0] if stairs else None
            
            if cur_stairs:
                for kpx, kpy, m in [(float(p[0][0]),float(p[0][1]),150),(float(p[11][0]),float(p[11][1]),200),(float(p[12][0]),float(p[12][1]),200),(float(p[15][0]),float(p[15][1]),150),(float(p[16][0]),float(p[16][1]),150)]:
                    if iv([kpx,kpy]) and point_in_box(kpx,kpy,cur_stairs,m): in_stairs=True; break
            if iv(p[0]):
                for f in fence:
                    if point_in_box(float(p[0][0]),float(p[0][1]),f,150): in_conv=True

            # CONVEYOR LOGIC
            if conv and in_conv and conv_mov:
                miss = get_missing_ppe(["hardhat","gloves","jacket","safety shoes","goggles"], ct, ppe_last_seen, PPE_MEMORY)
                if miss: log_to_csv("WARNING", f"CONVEYOR ZONE - MISSING: {', '.join(miss)}")
                
                for fx1,fy1,fx2,fy2 in fence:
                    cv2.rectangle(annotated, (int(fx1),int(fy1)),(int(fx2),int(fy2)), (255,255,0), 3)
                    
                    edge_margin = int((fx2 - fx1) * 0.15)
                    
                    cv2.line(annotated, (int(fx1 + edge_margin),int(fy1)),(int(fx1 + edge_margin),int(fy2)), (0, 165, 255), 2)
                    cv2.line(annotated, (int(fx2 - edge_margin),int(fy1)),(int(fx2 - edge_margin),int(fy2)), (0, 165, 255), 2)
                    
                    any_hand_crossed, any_hand_touching = False, False
                    for wx,wy in [(lwx,lwy),(rwx,rwy)]:
                        if (fx1 < wx < fx2) and (fy1 < wy < fy2):
                            if (fx1 + edge_margin) < wx < (fx2 - edge_margin):
                                any_hand_crossed = True
                            else:
                                any_hand_touching = True
                                
                    if any_hand_crossed:
                        crit.append("!! DON'T PUT HAND INSIDE FENCE !!"); log_to_csv("CRITICAL", "DON'T PUT HAND INSIDE FENCE")
                    elif any_hand_touching:
                        warn.append("DON'T TOUCH FENCE"); log_to_csv("WARNING", "DON'T TOUCH FENCE")

            # STAIRS LOGIC
            if in_stairs:
                if cur_stairs:
                    cv2.rectangle(annotated, (int(cur_stairs[0]),int(cur_stairs[1])),(int(cur_stairs[2]),int(cur_stairs[3])), (0,255,255), 2)
                for r in rails_raw: cv2.rectangle(annotated, (int(r[0]),int(r[1])),(int(r[2]),int(r[3])), (40,40,40), 10)
                
                up = check_phone_usage(p, phones)
                ws = check_carrying_weight(p, weights)
                mc, lc, det = check_three_point_contact_foolproof(p, rails_raw, cur_stairs)
                
                if not mc:
                    if ws["both_hands"]: crit.append("!! NO 3-POINT CONTACT - HANDS FULL !!"); log_to_csv("CRITICAL", "NO 3-POINT CONTACT - HANDS FULL")
                    else: warn.append(f"MAINTAIN 3-POINT CONTACT! ({lc}/3)"); log_to_csv("WARNING", f"MAINTAIN 3-POINT CONTACT ({lc}/3)")
                if up: crit.append("!! PHONE DETECTED ON STAIRS !!"); log_to_csv("CRITICAL", "PHONE DETECTED ON STAIRS")
                if ws["both_hands"]: crit.append("!! BOTH HANDS FULL ON STAIRS !!"); log_to_csv("CRITICAL", "BOTH HANDS FULL ON STAIRS")
                elif ws["one_hand"]: warn.append("CARRYING LOAD WITH ONE HAND"); log_to_csv("WARNING", "CARRYING LOAD WITH ONE HAND")
                if not crit and not warn and mc: safe.append("SAFE: 3-POINT CONTACT OK")

    # STEP 2: ZONE DETERMINATION
    if len(el_pan) > 0:
        last_electrical_seen = ct 
    is_electrical_zone = (ct - last_electrical_seen) < ELECTRICAL_ZONE_STICKY_TIME

    # STEP 3: EXCLUSIVE PPE ALERTS 
    if is_electrical_zone:
        for ep in el_pan:
            exp_ep = (ep[0]-250, ep[1]-200, ep[2]+250, ep[3]+200)
            zx1, zy1, zx2, zy2 = map(int, exp_ep)
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 165, 255), 2)
            cv2.putText(annotated, "ELECTRICAL ZONE", (zx1 + 5, zy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        wearing_wrong_ppe = ("jacket" in ppe_seen or "gloves" in ppe_seen)
        if wearing_wrong_ppe:
            crit.append("!! WEARING NORMAL PPE IN ELEC. ZONE !!")
            log_to_csv("CRITICAL", "WEARING NORMAL PPE IN ELECTRICAL ZONE")
            
    elif not in_conv and not in_stairs:
        gm = get_missing_ppe(["hardhat","jacket","safety shoes"], ct, ppe_last_seen, PPE_MEMORY)
        if gm: 
            log_to_csv("WARNING", f"MISSING PPE: {', '.join(gm)}")

    # STEP 4: DRAW UI
    yp = 15
    for a in crit: yp = draw_crit(annotated, a, yp)
    for a in warn: yp = draw_warn(annotated, a, yp)
    sy = frame_height - 80
    for a in safe: sy = draw_safe(annotated, a, sy)
    
    current_zone = "ELECTRICAL" if is_electrical_zone else "NORMAL"
    panel_x = int(annotated.shape[1]-385)
    panel_y = 10
    panel_w = 375
    panel_h = 320 
    cv2.rectangle(annotated, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (0,0,0), -1)
    
    draw_ppe_panel(annotated, ct, ppe_last_seen, PPE_MEMORY, zone=current_zone)
    
    out.write(annotated)
    cv2.imshow("Industrial Safety Monitoring", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done! Video is example_7.mp4. CSV has exactly 3 alerts for your PPT.")