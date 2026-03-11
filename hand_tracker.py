"""
hand_tracker.py
---------------
This module contains the HandTracker class, which wraps MediaPipe's
hand-detection solution into a simple, reusable interface.

How it works:
  1. MediaPipe processes each video frame and looks for hand shapes.
  2. If a hand is found, it returns 21 landmark points (x, y, z coordinates).
  3. We draw those landmarks and the skeleton connections on the frame.
  4. We also return the raw landmark data so other modules can use it
     (e.g., for gesture recognition or finger counting).

Landmark index reference (right hand, mirrored for left):
  0  = WRIST
  1–4   = THUMB  (CMC → MCP → IP → TIP)
  5–8   = INDEX  (MCP → PIP → DIP → TIP)
  9–12  = MIDDLE (MCP → PIP → DIP → TIP)
  13–16 = RING   (MCP → PIP → DIP → TIP)
  17–20 = PINKY  (MCP → PIP → DIP → TIP)
"""

import cv2
import mediapipe as mp


class HandTracker:
    """
    A clean wrapper around MediaPipe Hands for real-time hand tracking.

    Parameters
    ----------
    mode : bool
        False  → video mode  (faster, designed for streaming frames).
        True   → image mode  (slower, each frame treated independently).
    max_hands : int
        Maximum number of hands to detect simultaneously (1 or 2).
    detection_confidence : float
        Minimum confidence score (0.0–1.0) to consider a detection valid.
    tracking_confidence : float
        Minimum confidence score (0.0–1.0) to keep tracking across frames.
    """

    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # MediaPipe drawing utilities (for landmarks + connections)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # Custom drawing styles for a polished look
        self.landmark_style = self.mp_draw.DrawingSpec(
            color=(0, 255, 0),   # Green landmarks
            thickness=2,
            circle_radius=4,
        )
        self.connection_style = self.mp_draw.DrawingSpec(
            color=(255, 255, 255),  # White connections
            thickness=2,
        )

        # Initialise the MediaPipe Hands model
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

        # Will hold the latest detection results
        self.results = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_hands(self, frame, draw: bool = True):
        """
        Detect hands in a BGR frame and optionally draw landmarks.

        Parameters
        ----------
        frame : np.ndarray
            A single video frame in BGR colour format (as returned by OpenCV).
        draw : bool
            If True, landmark points and connections are drawn onto the frame.

        Returns
        -------
        frame : np.ndarray
            The (possibly annotated) frame.
        """
        # MediaPipe expects RGB, but OpenCV gives us BGR → convert
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the hand-detection model
        self.results = self.hands.process(rgb_frame)

        # If one or more hands were found, draw them
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.landmark_style,
                        self.connection_style,
                    )

        return frame

    def find_positions(self, frame, hand_index: int = 0):
        """
        Return the pixel positions of all 21 landmarks for one hand.

        Parameters
        ----------
        frame : np.ndarray
            The current video frame (used to convert normalised coords
            to pixel coords).
        hand_index : int
            Which hand to use if multiple hands are detected (0 = first hand).

        Returns
        -------
        landmark_list : list of [id, x, y]
            Each entry is [landmark_id (0-20), x_pixel, y_pixel].
            Returns an empty list if no hand is detected.
        """
        landmark_list = []
        height, width, _ = frame.shape

        if self.results and self.results.multi_hand_landmarks:
            # Make sure the requested hand index actually exists
            if hand_index < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_index]

                for landmark_id, lm in enumerate(hand.landmark):
                    # lm.x and lm.y are normalised (0.0 – 1.0);
                    # multiply by frame dimensions to get pixel coords
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    landmark_list.append([landmark_id, x_px, y_px])

        return landmark_list

    def get_hand_count(self) -> int:
        """Return the number of hands currently detected (0, 1, or 2)."""
        if self.results and self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0

    def get_handedness(self) -> list:
        """
        Return which hand(s) are detected.

        Returns
        -------
        list of str  →  e.g. ['Right'], ['Left'], or ['Right', 'Left']
        """
        labels = []
        if self.results and self.results.multi_handedness:
            for hand_info in self.results.multi_handedness:
                labels.append(hand_info.classification[0].label)
        return labels
