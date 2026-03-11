"""
main.py
-------
Entry point for the Real-Time Hand Gesture Recognition project.

What this script does:
  1. Opens your webcam.
  2. Passes each frame to HandTracker, which uses MediaPipe to detect
     hands and draw the 21 landmark points + skeleton connections.
  3. Overlays helpful information on the screen:
       • Frames Per Second (FPS)
       • Number of hands detected
       • Which hand(s) are visible (Left / Right)
       • Pixel coordinates of landmark #8 (index fingertip) as a demo
  4. Press  Q  to quit the programme cleanly.

Run with:
    python main.py
"""

import cv2
import numpy as np
import time

from hand_tracker import HandTracker


# ── Appearance constants ────────────────────────────────────────────────────
FONT            = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE     = (255, 255, 255)
COLOR_GREEN     = (0, 255, 0)
COLOR_CYAN      = (255, 255, 0)
COLOR_RED       = (0, 0, 255)
COLOR_OVERLAY   = (20, 20, 20)      # dark semi-transparent box background


def draw_info_panel(frame, fps: float, hand_count: int, handedness: list):
    """
    Draw a semi-transparent info panel in the top-left corner.

    Parameters
    ----------
    frame       : current video frame (modified in-place)
    fps         : current frames-per-second
    hand_count  : number of hands detected
    handedness  : list of hand labels, e.g. ['Right', 'Left']
    """
    # --- Semi-transparent background rectangle ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, 110), COLOR_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # --- FPS ---
    fps_color = COLOR_GREEN if fps >= 20 else COLOR_RED
    cv2.putText(frame, f"FPS : {int(fps)}", (20, 38),
                FONT, 0.7, fps_color, 2)

    # --- Hand count ---
    cv2.putText(frame, f"Hands: {hand_count}", (20, 66),
                FONT, 0.7, COLOR_WHITE, 2)

    # --- Handedness labels ---
    label_text = ", ".join(handedness) if handedness else "None"
    cv2.putText(frame, f"Type : {label_text}", (20, 94),
                FONT, 0.7, COLOR_CYAN, 2)


def draw_fingertip_coords(frame, landmark_list: list):
    """
    Highlight the index fingertip (landmark 8) and show its coordinates.
    This is a simple demo of using raw landmark data for custom logic.

    Parameters
    ----------
    frame         : current video frame (modified in-place)
    landmark_list : list of [id, x, y] from HandTracker.find_positions()
    """
    if not landmark_list:
        return

    # Landmark 8 is the INDEX fingertip
    for lm in landmark_list:
        if lm[0] == 8:
            x, y = lm[1], lm[2]

            # Draw a bright circle on the fingertip
            cv2.circle(frame, (x, y), 12, COLOR_CYAN, cv2.FILLED)
            cv2.circle(frame, (x, y), 14, COLOR_WHITE, 2)

            # Show coordinates just above the circle
            coord_text = f"({x}, {y})"
            cv2.putText(frame, coord_text, (x - 40, y - 20),
                        FONT, 0.55, COLOR_WHITE, 1)
            break


def main():
    # ── 1. Open Webcam ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)           # 0 = default webcam

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check that it is connected.")
        return

    # Optional: set a higher resolution for a clearer image
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Webcam opened successfully.")
    print("[INFO] Press  Q  to quit.\n")

    # ── 2. Initialise the Hand Tracker ────────────────────────────────────
    tracker = HandTracker(
        max_hands=2,
        detection_confidence=0.7,
        tracking_confidence=0.5,
    )

    # ── 3. FPS bookkeeping ────────────────────────────────────────────────
    prev_time = time.time()

    # ── 4. Main Loop ──────────────────────────────────────────────────────
    while True:
        success, frame = cap.read()

        if not success or frame is None:
            print("[WARNING] Failed to read frame from webcam. Retrying…")
            continue

        # Mirror the frame so it feels like a selfie-camera (more natural)
        frame = cv2.flip(frame, 1)

        # ── Detect hands and draw landmarks ──
        frame = tracker.find_hands(frame, draw=True)

        # ── Get landmark positions for hand 0 (first hand) ──
        landmark_list = tracker.find_positions(frame, hand_index=0)

        # ── Highlight the index fingertip if a hand is present ──
        draw_fingertip_coords(frame, landmark_list)

        # ── Calculate FPS ──
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)   # +tiny value avoids /0
        prev_time = curr_time

        # ── Draw the info panel ──
        draw_info_panel(
            frame,
            fps=fps,
            hand_count=tracker.get_hand_count(),
            handedness=tracker.get_handedness(),
        )

        # ── "Press Q to quit" reminder ──
        cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 12),
                    FONT, 0.5, (150, 150, 150), 1)

        # ── Show the frame ──
        cv2.imshow("Real-Time Hand Gesture Recognition", frame)

        # ── Quit on Q or Esc ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # 27 = Esc
            print("[INFO] Quitting…")
            break

    # ── 5. Clean Up ───────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    main()
