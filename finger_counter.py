"""
finger_counter.py
-----------------
Real-Time Finger Counter using Hand Landmark Detection.

How finger counting works:
  Each finger has a TIP landmark and a PIP (middle knuckle) landmark.
  If the TIP's y-position is ABOVE the PIP's y-position on screen,
  the finger is considered UP (open). Otherwise it's DOWN (closed).

  The THUMB is special — it moves sideways, so we compare X positions
  instead of Y positions.

Landmark IDs used:
  Thumb  : TIP=4,  compare with MCP=2  (x-axis)
  Index  : TIP=8,  compare with PIP=6  (y-axis)
  Middle : TIP=12, compare with PIP=10 (y-axis)
  Ring   : TIP=16, compare with PIP=14 (y-axis)
  Pinky  : TIP=20, compare with PIP=18 (y-axis)

Run with:
    py -3.11 finger_counter.py
"""

import cv2
import time
from hand_tracker import HandTracker

# ── Landmark IDs ────────────────────────────────────────────────────────────
# [TIP, PIP/MCP] for each finger
FINGER_LANDMARKS = {
    "Thumb":  [4, 2],
    "Index":  [8, 6],
    "Middle": [12, 10],
    "Ring":   [16, 14],
    "Pinky":  [20, 18],
}

# Finger tip IDs in order (used for counting)
TIP_IDS  = [4, 8, 12, 16, 20]
PIP_IDS  = [2, 6, 10, 14, 18]

# ── Display emoji for each count ────────────────────────────────────────────
COUNT_EMOJI = {
    0: "✊  FIST",
    1: "☝️  ONE",
    2: "✌️  TWO",
    3: "🤟  THREE",
    4: "🖖  FOUR",
    5: "🖐  FIVE",
    10: "🙌  TEN!",
}

# ── Colours ─────────────────────────────────────────────────────────────────
FONT          = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE   = (255, 255, 255)
COLOR_GREEN   = (0, 255, 0)
COLOR_CYAN    = (255, 255, 0)
COLOR_RED     = (0, 0, 255)
COLOR_ORANGE  = (0, 165, 255)
COLOR_DARK    = (20, 20, 20)
COLOR_YELLOW  = (0, 215, 255)


# ── Helper: count fingers ────────────────────────────────────────────────────

def count_fingers(landmark_list: list, handedness: str) -> tuple[int, list]:
    """
    Count how many fingers are up for one hand.

    Parameters
    ----------
    landmark_list : list of [id, x, y]
        21 landmark positions from HandTracker.find_positions()
    handedness : str
        'Right' or 'Left' — needed to correctly judge the thumb direction.

    Returns
    -------
    total : int
        Number of fingers that are up (0–5).
    fingers : list of bool
        [thumb, index, middle, ring, pinky] — True = up, False = down.
    """
    if not landmark_list:
        return 0, []

    # Build a quick lookup: {landmark_id: (x, y)}
    lm = {item[0]: (item[1], item[2]) for item in landmark_list}

    fingers = []

    # ── Thumb (compare x-axis) ──────────────────────────────────────────
    # For a RIGHT hand (mirrored): thumb tip is to the LEFT of thumb MCP when open
    # For a LEFT  hand (mirrored): thumb tip is to the RIGHT of thumb MCP when open
    if handedness == "Right":
        thumb_up = lm[4][0] < lm[2][0]   # tip.x < mcp.x → open
    else:
        thumb_up = lm[4][0] > lm[2][0]   # tip.x > mcp.x → open
    fingers.append(thumb_up)

    # ── Four fingers (compare y-axis) ───────────────────────────────────
    # tip.y < pip.y means the tip is HIGHER on screen → finger is UP
    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        fingers.append(lm[tip_id][1] < lm[pip_id][1])

    total = sum(fingers)
    return total, fingers


# ── Helper: draw finger status bar ──────────────────────────────────────────

def draw_finger_status(frame, fingers: list, start_x: int, start_y: int):
    """
    Draw a small row of finger indicators (filled = up, hollow = down).
    """
    if not fingers:
        return

    labels = ["T", "I", "M", "R", "P"]   # Thumb, Index, Middle, Ring, Pinky
    for i, (label, is_up) in enumerate(zip(labels, fingers)):
        x = start_x + i * 44
        color = COLOR_GREEN if is_up else (80, 80, 80)
        # Circle: filled if up, hollow if down
        if is_up:
            cv2.circle(frame, (x, start_y), 18, color, -1)
            cv2.circle(frame, (x, start_y), 18, COLOR_WHITE, 1)
        else:
            cv2.circle(frame, (x, start_y), 18, (40, 40, 40), -1)
            cv2.circle(frame, (x, start_y), 18, (100, 100, 100), 1)
        cv2.putText(frame, label, (x - 6, start_y + 5),
                    FONT, 0.45, COLOR_WHITE, 1)


# ── Helper: draw big count display ──────────────────────────────────────────

def draw_count_display(frame, total: int, h: int, w: int):
    """
    Draw a large centred number showing total fingers up.
    Also shows the emoji label below it.
    """
    # Big number — shadow + main
    num_text = str(total)
    num_scale = 5.0
    thickness = 12

    (tw, th), _ = cv2.getTextSize(num_text, FONT, num_scale, thickness)
    cx = (w - tw) // 2
    cy = h // 2 + th // 2 - 40

    # Shadow
    cv2.putText(frame, num_text, (cx + 4, cy + 4),
                FONT, num_scale, (0, 0, 0), thickness + 4)
    # Coloured number
    number_color = COLOR_CYAN if total > 0 else COLOR_ORANGE
    cv2.putText(frame, num_text, (cx, cy),
                FONT, num_scale, number_color, thickness)

    # Emoji label below
    label = COUNT_EMOJI.get(total, f"{total} FINGERS")
    (lw, _), _ = cv2.getTextSize(label, FONT, 0.9, 2)
    lx = (w - lw) // 2
    cv2.putText(frame, label, (lx, cy + 55),
                FONT, 0.9, COLOR_YELLOW, 2)


# ── Helper: draw info panel ─────────────────────────────────────────────────

def draw_info_panel(frame, fps: float, hand_count: int, handedness: list):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (270, 115), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    fps_color = COLOR_GREEN if fps >= 20 else COLOR_RED
    cv2.putText(frame, f"FPS   : {int(fps)}", (20, 38),  FONT, 0.65, fps_color, 2)
    cv2.putText(frame, f"Hands : {hand_count}",  (20, 66),  FONT, 0.65, COLOR_WHITE, 2)
    label = ", ".join(handedness) if handedness else "None"
    cv2.putText(frame, f"Type  : {label}", (20, 94),  FONT, 0.65, COLOR_CYAN, 2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Finger Counter started.")
    print("[INFO] Hold your hand up in front of the camera.")
    print("[INFO] Press Q or Esc to quit.\n")

    tracker   = HandTracker(max_hands=2, detection_confidence=0.7)
    prev_time = time.time()

    # Track combined total when 2 hands visible
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Detect hands ──
        frame = tracker.find_hands(frame, draw=True)

        hand_count  = tracker.get_hand_count()
        handedness  = tracker.get_handedness()

        grand_total = 0
        all_fingers = []

        # ── Count fingers for each hand ──
        for i in range(hand_count):
            lm_list  = tracker.find_positions(frame, hand_index=i)
            side     = handedness[i] if i < len(handedness) else "Right"
            total, fingers = count_fingers(lm_list, side)
            grand_total += total
            all_fingers  = fingers   # show last hand's individual fingers

            # Label each hand with its count
            if lm_list:
                wrist_x = lm_list[0][1]
                wrist_y = lm_list[0][2]
                cv2.putText(frame, f"{side}: {total}",
                            (wrist_x - 30, wrist_y + 40),
                            FONT, 0.8, COLOR_ORANGE, 2)

        # ── Draw big number in centre ──
        if hand_count > 0:
            draw_count_display(frame, grand_total, h, w)

            # Finger status dots (bottom centre)
            dot_start_x = w // 2 - 88
            dot_start_y = h - 50
            draw_finger_status(frame, all_fingers, dot_start_x, dot_start_y)

        else:
            # No hand — show prompt
            msg = "Show your hand to the camera!"
            (mw, _), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
            cv2.putText(frame, msg, ((w - mw) // 2, h // 2),
                        FONT, 0.9, (150, 150, 150), 2)

        # ── FPS ──
        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        # ── Info panel ──
        draw_info_panel(frame, fps, hand_count, handedness)

        # ── Quit hint ──
        cv2.putText(frame, "Press Q to quit", (10, h - 12),
                    FONT, 0.5, (120, 120, 120), 1)

        cv2.imshow("Finger Counter AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("[INFO] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
