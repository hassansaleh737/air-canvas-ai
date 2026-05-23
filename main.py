import cv2
import os
import time
import numpy as np

from hand_tracker import HandTracker
from drawing_canvas import DrawingCanvas
from config import (
    CAMERA_INDEX, WINDOW_NAME,
    DEFAULT_LEFT_HAND_COLOR, DEFAULT_RIGHT_HAND_COLOR,
    BRUSH_THICKNESS, ERASER_THICKNESS,
    RED, BLUE, GREEN, YELLOW, BLACK, WHITE,
    SAVE_PATH, MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE
)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ══════════════════════════════════════════════════════════════════
#  THEME  —  all values are BGR
# ══════════════════════════════════════════════════════════════════
T = {
    "panel":        (30, 24, 22),
    "panel_mid":    (50, 42, 38),
    "border":       (85, 75, 68),
    "border_hi":    (130, 110, 96),
    "text":         (245, 245, 245),
    "muted":        (150, 145, 138),
    "dim":          (82, 79, 76),
    "purple":       (224, 72, 96),
    "purple_soft":  (255, 147, 168),
    "green":        (117, 200, 34),
    "lhand":        (102, 80, 255),
    "rhand":        (255, 130, 70),
    "disabled":     (80, 80, 78),
}

TOP_H    = 64
BOT_H    = 56
CARD_W   = 134
CARD_H   = 62
CARD_GAP = 8
CARD_PAD = 14

SHAPE_MODES = ["Free Draw", "Line", "Rectangle", "Circle", "Triangle"]


# ══════════════════════════════════════════════════════════════════
#  LOW-LEVEL DRAWING PRIMITIVES
# ══════════════════════════════════════════════════════════════════

def rrect(img, x1, y1, x2, y2, r, color, thick=-1):
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))

    if thick == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)

        for cx, cy in [
            (x1 + r, y1 + r),
            (x2 - r, y1 + r),
            (x1 + r, y2 - r),
            (x2 - r, y2 - r)
        ]:
            cv2.circle(img, (cx, cy), r, color, -1)

    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thick)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thick)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thick)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thick)

        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, thick)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, thick)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thick)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, thick)


def txt(img, text, x, y, scale=0.50, color=None, bold=False):
    cv2.putText(
        img,
        text,
        (int(x), int(y)),
        FONT,
        scale,
        color if color else T["text"],
        2 if bold else 1,
        cv2.LINE_AA
    )


def tsize(text, scale=0.50, bold=False):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 2 if bold else 1)
    return w, h


def alpha_fill(img, x1, y1, x2, y2, color, alpha):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def alpha_rrect(img, x1, y1, x2, y2, r, color, alpha):
    ov = img.copy()
    rrect(ov, x1, y1, x2, y2, r, color)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def pill(img, text, x, y, h, bg, fg, scale=0.40):
    tw, th = tsize(text, scale)
    pad = 10
    x2 = x + tw + pad * 2
    y2 = y + h

    rrect(img, x, y, x2, y2, 5, bg)
    txt(img, text, x + pad, y + (h + th) // 2, scale, fg)

    return x2


def vdiv(img, x, y1, y2, color=None):
    cv2.line(img, (x, y1), (x, y2), color if color else T["border"], 1)


# ══════════════════════════════════════════════════════════════════
#  LOGIC HELPERS
# ══════════════════════════════════════════════════════════════════

def get_selected_text(sel):
    return {
        "Left": "Left Hand",
        "Right": "Right Hand"
    }.get(sel, "Both Hands")


def get_hand_key_by_screen_position(frame, landmarks):
    _, w, _ = frame.shape
    return "Left" if landmarks[0][1] < w // 2 else "Right"


def apply_color_to_selected(hand_settings, sel, color, thickness):
    targets = ["Left", "Right"] if sel == "Both" else [sel]

    for k in targets:
        hand_settings[k]["color"] = color
        hand_settings[k]["thickness"] = thickness


def apply_brush_size_to_selected(hand_settings, sel, brush_size):
    targets = ["Left", "Right"] if sel == "Both" else [sel]

    for k in targets:
        hand_settings[k]["thickness"] = brush_size


def toggle_selected_hand(hand_settings, sel):
    if sel == "Both":
        new = not (
            hand_settings["Left"]["active"] or
            hand_settings["Right"]["active"]
        )

        hand_settings["Left"]["active"] = new
        hand_settings["Right"]["active"] = new

    else:
        hand_settings[sel]["active"] = not hand_settings[sel]["active"]


def next_shape_mode(current_mode):
    idx = SHAPE_MODES.index(current_mode)
    return SHAPE_MODES[(idx + 1) % len(SHAPE_MODES)]


# ══════════════════════════════════════════════════════════════════
#  LAYER HELPERS + SAVE
# ══════════════════════════════════════════════════════════════════

def has_pixels(img):
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.count_nonzero(gray) > 0


def overlay_layer(base_img, layer_img):
    if layer_img is None:
        return base_img

    gray = cv2.cvtColor(layer_img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    base_img[mask] = layer_img[mask]
    return base_img


def make_unique_save_path(base_path):
    folder = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)

    if folder == "":
        folder = "output"

    os.makedirs(folder, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    new_path = os.path.join(folder, f"{name}_{timestamp}{ext}")

    counter = 1
    while os.path.exists(new_path):
        new_path = os.path.join(folder, f"{name}_{timestamp}_{counter}{ext}")
        counter += 1

    return new_path


def save_final_canvas(canvas, base_path, writing_layer=None):
    if canvas.canvas is None and writing_layer is None:
        return False, None

    if canvas.canvas is not None:
        final_img = canvas.canvas.copy()
    elif writing_layer is not None:
        final_img = np.zeros_like(writing_layer)
    else:
        return False, None

    if writing_layer is not None and has_pixels(writing_layer):
        final_img = overlay_layer(final_img, writing_layer)

    if not has_pixels(final_img):
        return False, None

    unique_path = make_unique_save_path(base_path)
    cv2.imwrite(unique_path, final_img)
    return True, unique_path


# ══════════════════════════════════════════════════════════════════
#  CLEAN HANDWRITING STROKE HELPERS
#  مفيش OCR هنا — الكتابة بتفضل بإيدك، بس بتتحفظ أنعم وأنضف
# ══════════════════════════════════════════════════════════════════

def smooth_points(points, factor=0.35):
    if len(points) < 3:
        return points

    smoothed = [points[0]]

    for i in range(1, len(points)):
        prev_x, prev_y = smoothed[-1]
        cur_x, cur_y = points[i]

        new_x = int(prev_x * (1 - factor) + cur_x * factor)
        new_y = int(prev_y * (1 - factor) + cur_y * factor)

        smoothed.append((new_x, new_y))

    return smoothed


def reduce_points_distance(points, min_distance=3):
    if len(points) < 2:
        return points

    reduced = [points[0]]

    for p in points[1:]:
        if np.linalg.norm(np.array(p) - np.array(reduced[-1])) >= min_distance:
            reduced.append(p)

    return reduced


def draw_smooth_stroke(layer, points, color, thickness):
    if layer is None or len(points) < 2:
        return

    clean_points = reduce_points_distance(points, min_distance=3)
    clean_points = smooth_points(clean_points, factor=0.32)

    if len(clean_points) < 2:
        return

    thickness = max(2, int(thickness))

    for i in range(1, len(clean_points)):
        cv2.line(
            layer,
            clean_points[i - 1],
            clean_points[i],
            color,
            thickness,
            cv2.LINE_AA
        )

    # Soft round caps, so handwritten characters look cleaner
    for p in [clean_points[0], clean_points[-1]]:
        cv2.circle(layer, p, max(1, thickness // 2), color, -1, cv2.LINE_AA)


def preview_smooth_stroke(frame, points, color, thickness):
    if len(points) < 2:
        return

    temp = np.zeros_like(frame)
    draw_smooth_stroke(temp, points, color, thickness)
    overlay_layer(frame, temp)


def commit_writing_stroke(writing_layer, hand_key, writing_strokes):
    stroke = writing_strokes.get(hand_key)

    if stroke is None:
        return False

    points = stroke.get("points", [])
    color = stroke.get("color", WHITE)
    thickness = stroke.get("thickness", BRUSH_THICKNESS)

    if len(points) >= 2:
        draw_smooth_stroke(writing_layer, points, color, thickness)

    writing_strokes[hand_key] = None
    return True


def commit_all_writing_strokes(writing_layer, writing_strokes):
    for hand_key in list(writing_strokes.keys()):
        commit_writing_stroke(writing_layer, hand_key, writing_strokes)


# ══════════════════════════════════════════════════════════════════
#  SHAPE MODE HELPERS
# ══════════════════════════════════════════════════════════════════

def point_distance(p1, p2):
    if p1 is None or p2 is None:
        return 0

    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def triangle_points_from_box(start, end):
    x1, y1 = start
    x2, y2 = end

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    cx = (left + right) // 2

    pts = np.array([
        [cx, top],
        [left, bottom],
        [right, bottom]
    ], dtype=np.int32)

    return pts.reshape((-1, 1, 2))


def draw_clean_shape(img, mode, start, end, color, thickness):
    if start is None or end is None:
        return

    if point_distance(start, end) < 8:
        return

    x1, y1 = start
    x2, y2 = end

    if mode == "Line":
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    elif mode == "Rectangle":
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    elif mode == "Circle":
        radius = int(point_distance(start, end))
        if radius > 4:
            cv2.circle(img, (x1, y1), radius, color, thickness, cv2.LINE_AA)

    elif mode == "Triangle":
        pts = triangle_points_from_box(start, end)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def draw_shape_preview(frame, mode, start, end, color, thickness):
    if mode == "Free Draw":
        return

    preview = frame.copy()
    draw_clean_shape(preview, mode, start, end, color, max(2, thickness))
    cv2.addWeighted(preview, 0.65, frame, 0.35, 0, frame)


def commit_shape_to_canvas(canvas, mode, start, end, color, thickness):
    if canvas.canvas is None:
        return False

    if mode == "Free Draw":
        return False

    if start is None or end is None:
        return False

    if point_distance(start, end) < 12:
        return False

    draw_clean_shape(canvas.canvas, mode, start, end, color, thickness)
    return True


def finish_shape_for_hand(canvas, hand_key, shape_mode, shape_states):
    state = shape_states.get(hand_key)

    if state is None:
        return False

    start = state.get("start")
    end = state.get("end")
    color = state.get("color")
    thickness = state.get("thickness")

    committed = commit_shape_to_canvas(canvas, shape_mode, start, end, color, thickness)
    shape_states[hand_key] = None
    canvas.reset_previous_point(hand_key)
    return committed


def finish_all_shapes(canvas, shape_mode, shape_states):
    for hand_key in list(shape_states.keys()):
        finish_shape_for_hand(canvas, hand_key, shape_mode, shape_states)


# ══════════════════════════════════════════════════════════════════
#  UI COMPONENTS — unchanged
# ══════════════════════════════════════════════════════════════════

def draw_top_bar(img, mode_text, fps, session_secs):
    _, w, _ = img.shape

    alpha_fill(img, 0, 0, w, TOP_H, T["panel"], 0.88)
    cv2.line(img, (0, TOP_H), (w, TOP_H), T["border"], 1)

    lx1, ly1 = 14, 11
    rrect(img, lx1, ly1, lx1 + 42, ly1 + 42, 8, T["purple"])
    ic = (lx1 + 21, ly1 + 21)
    cv2.circle(img, ic, 11, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, ic, 4, (255, 255, 255), -1)

    txt(img, "AIR  CANVAS", 68, 33, 0.60, T["text"], bold=True)
    txt(img, "Two-hand vision drawing", 68, 52, 0.37, T["muted"])

    vdiv(img, 295, 14, TOP_H - 14)

    bx = 308
    badge_y, badge_h = 18, 28

    lv_x1 = bx
    bx = pill(img, "       LIVE", bx, badge_y, badge_h, (22, 68, 32), T["green"], 0.38) + 8
    blink = 0.40 + 0.60 * abs(np.sin(time.time() * 2.8))
    dot_c = tuple(int(c * blink) for c in T["green"])
    cv2.circle(img, (lv_x1 + 14, badge_y + badge_h // 2), 4, dot_c, -1, cv2.LINE_AA)

    bx = pill(img, f"FPS  {fps:5.1f}", bx, badge_y, badge_h, T["panel_mid"], T["muted"], 0.38) + 8
    bx = pill(img, f"MODE  {mode_text.upper()}", bx, badge_y, badge_h, (50, 25, 40), T["purple_soft"], 0.38) + 8

    se = int(session_secs)
    hh, rem = divmod(se, 3600)
    mm, ss = divmod(rem, 60)
    ts = f"SESSION  {hh:02d}:{mm:02d}:{ss:02d}"
    tw, _ = tsize(ts, 0.35)
    txt(img, ts, w - tw - 18, 36, 0.35, T["dim"])


def draw_hand_panels(img, hand_settings, sel):
    _, w, _ = img.shape

    r_edge = w - CARD_PAD
    lc_x1 = r_edge - CARD_W * 2 - CARD_GAP
    lc_x2 = lc_x1 + CARD_W
    rc_x1 = lc_x2 + CARD_GAP
    rc_x2 = r_edge
    cy1 = TOP_H + 8
    cy2 = cy1 + CARD_H

    cards = [
        ("Left", "LEFT HAND", T["lhand"], lc_x1, lc_x2),
        ("Right", "RIGHT HAND", T["rhand"], rc_x1, rc_x2),
    ]

    for key, label, hue, cx1, cx2 in cards:
        active = hand_settings[key]["active"]
        selected = sel in (key, "Both")
        dot_c = hue if active else T["disabled"]
        bdr_c = hue if selected else T["border"]
        bdr_th = 2 if selected else 1

        alpha_fill(img, cx1, cy1, cx2, cy2, T["panel_mid"], 0.85)
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), bdr_c, bdr_th)

        dot_x, dot_y = cx1 + 20, cy1 + 22
        cv2.circle(img, (dot_x, dot_y), 6, dot_c, -1, cv2.LINE_AA)

        if active:
            cv2.circle(img, (dot_x, dot_y), 9, dot_c, 1, cv2.LINE_AA)

        txt(img, label, cx1 + 33, cy1 + 26, 0.42, T["text"], bold=True)

        status_c = T["green"] if active else T["disabled"]
        txt(img, "ACTIVE" if active else "OFF", cx1 + 33, cy1 + 46, 0.36, status_c)

        tw_b, _ = tsize("BRUSH", 0.30)
        pill_x = cx2 - tw_b - 26
        pill(img, "BRUSH", pill_x, cy1 + 36, 18, T["panel"], T["dim"], 0.30)

        bar_y = cy2 - 7
        cv2.rectangle(img, (cx1 + 10, bar_y), (cx2 - 10, bar_y + 3), T["panel"], -1)

        if active:
            fill = int((cx2 - cx1 - 20) * 0.80)
            cv2.rectangle(img, (cx1 + 10, bar_y), (cx1 + 10 + fill, bar_y + 3), dot_c, -1)


def draw_bottom_bar(img, sel, hand_settings):
    h_img, w, _ = img.shape
    by = h_img - BOT_H
    cy = by + BOT_H // 2

    alpha_fill(img, 0, by, w, h_img, T["panel"], 0.90)
    cv2.line(img, (0, by), (w, by), T["border"], 1)

    bx = 14

    for label, key in [("L", "Left"), ("R", "Right"), ("B", "Both")]:
        active = sel == key
        bg = T["purple"] if active else T["panel_mid"]
        fg = T["text"] if active else T["muted"]
        btn_w = 30

        rrect(img, bx, cy - 14, bx + btn_w, cy + 14, 5, bg)

        if not active:
            cv2.rectangle(img, (bx, cy - 14), (bx + btn_w, cy + 14), T["border"], 1)

        tw, th = tsize(label, 0.40, bold=active)
        txt(img, label, bx + (btn_w - tw) // 2, cy + th // 2 + 1, 0.40, fg, bold=active)
        bx += btn_w + 4

    bx += 6
    vdiv(img, bx, by + 14, h_img - 14)
    bx += 12

    bi_text = "Index Finger = BRUSH"
    tw_bi, _ = tsize(bi_text, 0.38)
    bi_x2 = bx + tw_bi + 24

    rrect(img, bx, cy - 14, bi_x2, cy + 14, 5, T["panel_mid"])
    cv2.rectangle(img, (bx, cy - 14), (bi_x2, cy + 14), T["border"], 1)
    txt(img, bi_text, bx + 12, cy + 5, 0.38, T["muted"])

    bx = bi_x2 + 14

    for key, label in [("X", "Toggle"), ("C", "Clear"), ("S", "Save"), ("Q", "Quit")]:
        kw, kh = tsize(key, 0.36)
        box_w = kw + 10
        rrect(img, bx, cy - 11, bx + box_w, cy + 11, 3, T["panel_mid"])
        cv2.rectangle(img, (bx, cy - 11), (bx + box_w, cy + 11), T["border"], 1)
        txt(img, key, bx + (box_w - kw) // 2, cy + kh // 2 + 1, 0.36, T["muted"])
        bx += box_w + 6
        lw, _ = tsize(label, 0.35)
        txt(img, label, bx, cy + 5, 0.35, T["dim"])
        bx += lw + 14

    palette = [("1", RED), ("2", BLUE), ("3", GREEN), ("4", YELLOW), ("E", BLACK)]
    sw_r = 10
    sw_sp = 8
    pal_w = len(palette) * (sw_r * 2) + (len(palette) - 1) * sw_sp
    px = w - CARD_PAD - pal_w

    for pkey, color in palette:
        sx = px + sw_r
        cv2.circle(img, (sx, cy - 4), sw_r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (sx, cy - 4), sw_r + 2, (185, 180, 175), 1, cv2.LINE_AA)
        kw, _ = tsize(pkey, 0.28)
        txt(img, pkey, sx - kw // 2, cy + 14, 0.28, T["dim"])
        px += sw_r * 2 + sw_sp


def draw_pointer(img, x, y, hand_key, is_active, color_value):
    c = color_value if is_active else T["disabled"]

    cv2.circle(img, (x, y), 4, c, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 14, c, 2, cv2.LINE_AA)

    label = f"{hand_key}  {'ON' if is_active else 'OFF'}"
    tw, th = tsize(label, 0.36)
    lx, ly = x + 18, y - 30

    alpha_rrect(img, lx, ly, lx + tw + 16, ly + th + 14, 4, T["panel"], 0.88)
    cv2.rectangle(img, (lx, ly), (lx + tw + 16, ly + th + 14), c, 1)
    txt(img, label, lx + 8, ly + th + 8, 0.36, T["text"])


def draw_notification(img, message, timestamp):
    elapsed = time.time() - timestamp

    if not message or elapsed > 2.2:
        return

    h_img, w, _ = img.shape
    fade = 1.0 if elapsed < 1.4 else max(0.0, 1.0 - (elapsed - 1.4) / 0.8)

    tw, th = tsize(message, 0.46, bold=True)
    nw, nh = tw + 58, 38
    nx1 = w // 2 - nw // 2
    ny1 = h_img - BOT_H - nh - 14
    nx2, ny2 = nx1 + nw, ny1 + nh

    ov = img.copy()
    rrect(ov, nx1, ny1, nx2, ny2, 6, T["panel_mid"])
    cv2.rectangle(ov, (nx1, ny1), (nx2, ny2), T["border"], 1)
    cv2.circle(ov, (nx1 + 18, ny1 + nh // 2), 4, T["green"], -1, cv2.LINE_AA)
    txt(ov, message, nx1 + 32, ny1 + nh // 2 + 7, 0.46, T["text"], bold=True)
    cv2.addWeighted(ov, fade * 0.92, img, 1 - fade * 0.92, 0, img)


def draw_ui(img, mode_text, sel, hand_settings, fps, session_secs, notification, notif_time):
    draw_top_bar(img, mode_text, fps, session_secs)
    draw_hand_panels(img, hand_settings, sel)
    draw_bottom_bar(img, sel, hand_settings)
    draw_notification(img, notification, notif_time)


# ══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main():
    os.makedirs("output", exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(
        max_hands=MAX_HANDS,
        detection_confidence=DETECTION_CONFIDENCE,
        tracking_confidence=TRACKING_CONFIDENCE
    )

    canvas = DrawingCanvas()

    hand_settings = {
        "Left": {
            "color": DEFAULT_LEFT_HAND_COLOR,
            "thickness": BRUSH_THICKNESS,
            "active": True
        },
        "Right": {
            "color": DEFAULT_RIGHT_HAND_COLOR,
            "thickness": BRUSH_THICKNESS,
            "active": True
        },
    }

    sel = "Both"
    mode_text = "Ready"
    drawing_mode = "Free Draw"
    handwriting_mode = False
    brush_size = BRUSH_THICKNESS

    shape_states = {
        "Left": None,
        "Right": None
    }

    writing_layer = None
    writing_strokes = {
        "Left": None,
        "Right": None
    }

    last_t = time.time()
    session_start = time.time()
    fps = 0.0
    notification = "System ready"
    notif_time = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Error: Failed to read from camera.")
            break

        now = time.time()
        fps = 1.0 / max(now - last_t, 1e-4)
        last_t = now
        session_s = now - session_start

        frame = cv2.flip(frame, 1)
        canvas.initialize_canvas(frame)

        if writing_layer is None:
            writing_layer = np.zeros_like(frame)

        frame, results = tracker.find_hands(frame, draw=True)

        detected = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = tracker.get_landmark_positions(frame, hand_landmarks)
                fingers = tracker.fingers_up(landmarks)
                tip = tracker.get_index_finger_tip(landmarks)

                hk = get_hand_key_by_screen_position(frame, landmarks)
                detected.append(hk)

                col = hand_settings[hk]["color"]
                thk = hand_settings[hk]["thickness"]
                actv = hand_settings[hk]["active"]

                if tip:
                    x, y = tip
                    draw_pointer(frame, x, y, hk, actv, col)

                    if not actv:
                        mode_text = f"{hk} Off"
                        finish_shape_for_hand(canvas, hk, drawing_mode, shape_states)
                        commit_writing_stroke(writing_layer, hk, writing_strokes)
                        canvas.reset_previous_point(hk)
                        continue

                    # ─────────────────────────────
                    # Handwriting Mode
                    # تكتب بإيدك جوه البرنامج، ولما تتحفظ بتتخزن بخط أنعم
                    # ─────────────────────────────
                    if handwriting_mode:
                        mode_text = "Writing"

                        if fingers[1] == 1 and fingers[2] == 0:
                            write_color = col if col != BLACK else WHITE
                            write_thickness = max(2, thk)

                            if writing_strokes[hk] is None:
                                writing_strokes[hk] = {
                                    "points": [(x, y)],
                                    "color": write_color,
                                    "thickness": write_thickness
                                }
                            else:
                                writing_strokes[hk]["points"].append((x, y))
                                writing_strokes[hk]["color"] = write_color
                                writing_strokes[hk]["thickness"] = write_thickness

                            # Preview only — committed layer stays clean
                            preview_smooth_stroke(
                                frame,
                                writing_strokes[hk]["points"],
                                write_color,
                                write_thickness
                            )

                            canvas.reset_previous_point(hk)

                        elif fingers[1] == 1 and fingers[2] == 1:
                            mode_text = "Move"
                            commit_writing_stroke(writing_layer, hk, writing_strokes)
                            canvas.reset_previous_point(hk)

                        else:
                            mode_text = "Writing Idle"
                            commit_writing_stroke(writing_layer, hk, writing_strokes)
                            canvas.reset_previous_point(hk)

                        continue

                    # ─────────────────────────────
                    # Free Draw Mode
                    # ─────────────────────────────
                    if drawing_mode == "Free Draw":
                        if fingers[1] == 1 and fingers[2] == 0:
                            mode_text = "Drawing"
                            canvas.draw_line(hk, x, y, col, thk)

                        elif fingers[1] == 1 and fingers[2] == 1:
                            mode_text = "Move"
                            canvas.reset_previous_point(hk)

                        else:
                            mode_text = "Idle"
                            canvas.reset_previous_point(hk)

                    # ─────────────────────────────
                    # Shape Modes
                    # ─────────────────────────────
                    else:
                        if fingers[1] == 1 and fingers[2] == 0:
                            mode_text = drawing_mode

                            if shape_states[hk] is None:
                                shape_states[hk] = {
                                    "start": (x, y),
                                    "end": (x, y),
                                    "color": col,
                                    "thickness": thk
                                }
                            else:
                                shape_states[hk]["end"] = (x, y)
                                shape_states[hk]["color"] = col
                                shape_states[hk]["thickness"] = thk

                            draw_shape_preview(
                                frame,
                                drawing_mode,
                                shape_states[hk]["start"],
                                shape_states[hk]["end"],
                                col,
                                thk
                            )

                            canvas.reset_previous_point(hk)

                        elif fingers[1] == 1 and fingers[2] == 1:
                            mode_text = "Move"
                            finish_shape_for_hand(canvas, hk, drawing_mode, shape_states)
                            canvas.reset_previous_point(hk)

                        else:
                            mode_text = "Idle"
                            finish_shape_for_hand(canvas, hk, drawing_mode, shape_states)
                            canvas.reset_previous_point(hk)

            for k in list(canvas.prev_points.keys()):
                if k not in detected:
                    canvas.reset_previous_point(k)

            for hand_key in ["Left", "Right"]:
                if hand_key not in detected:
                    finish_shape_for_hand(canvas, hand_key, drawing_mode, shape_states)
                    commit_writing_stroke(writing_layer, hand_key, writing_strokes)

        else:
            mode_text = "No Hand"
            finish_all_shapes(canvas, drawing_mode, shape_states)
            commit_all_writing_strokes(writing_layer, writing_strokes)
            canvas.reset_all_previous_points()

        frame = canvas.merge_with_frame(frame)
        frame = overlay_layer(frame, writing_layer)

        draw_ui(
            frame,
            mode_text,
            sel,
            hand_settings,
            fps,
            session_s,
            notification,
            notif_time
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        # ── keyboard controls
        if key == ord("q"):
            break

        elif key == ord("l"):
            sel = "Left"
            notification = "Selected Left Hand"
            notif_time = time.time()
            print(notification)

        elif key == ord("r"):
            sel = "Right"
            notification = "Selected Right Hand"
            notif_time = time.time()
            print(notification)

        elif key == ord("b"):
            sel = "Both"
            notification = "Selected Both Hands"
            notif_time = time.time()
            print(notification)

        elif key == ord("x"):
            toggle_selected_hand(hand_settings, sel)
            notification = f"Toggled  {get_selected_text(sel)}"
            notif_time = time.time()
            print(notification)

        elif key == ord("m"):
            handwriting_mode = False
            commit_all_writing_strokes(writing_layer, writing_strokes)
            finish_all_shapes(canvas, drawing_mode, shape_states)
            drawing_mode = next_shape_mode(drawing_mode)
            notification = f"Mode  →  {drawing_mode}"
            notif_time = time.time()
            print(notification)

        elif key == ord("t"):
            finish_all_shapes(canvas, drawing_mode, shape_states)
            handwriting_mode = not handwriting_mode

            if handwriting_mode:
                notification = "Handwriting mode ON"
            else:
                commit_all_writing_strokes(writing_layer, writing_strokes)
                notification = "Handwriting mode OFF"

            notif_time = time.time()
            print(notification)

        elif key in [ord("+"), ord("=")]:
            brush_size = min(30, brush_size + 2)
            apply_brush_size_to_selected(hand_settings, sel, brush_size)
            notification = f"Brush size  →  {brush_size}"
            notif_time = time.time()
            print(notification)

        elif key in [ord("-"), ord("_")]:
            brush_size = max(2, brush_size - 2)
            apply_brush_size_to_selected(hand_settings, sel, brush_size)
            notification = f"Brush size  →  {brush_size}"
            notif_time = time.time()
            print(notification)

        elif key == ord("c"):
            canvas.clear_canvas()
            finish_all_shapes(canvas, drawing_mode, shape_states)

            if writing_layer is not None:
                writing_layer = np.zeros_like(writing_layer)

            writing_strokes = {
                "Left": None,
                "Right": None
            }

            notification = "Canvas cleared"
            notif_time = time.time()
            print(notification)

        elif key == ord("s"):
            finish_all_shapes(canvas, drawing_mode, shape_states)
            commit_all_writing_strokes(writing_layer, writing_strokes)
            saved, saved_path = save_final_canvas(canvas, SAVE_PATH, writing_layer)

            if saved:
                notification = f"Saved: {os.path.basename(saved_path)}"
                print(f"Drawing saved to: {saved_path}")
            else:
                notification = "Nothing to save"
                print(notification)

            notif_time = time.time()

        elif key == ord("1"):
            apply_color_to_selected(hand_settings, sel, RED, brush_size)
            notification = f"Red  →  {get_selected_text(sel)}"
            notif_time = time.time()
            print(notification)

        elif key == ord("2"):
            apply_color_to_selected(hand_settings, sel, BLUE, brush_size)
            notification = f"Blue  →  {get_selected_text(sel)}"
            notif_time = time.time()
            print(notification)

        elif key == ord("3"):
            apply_color_to_selected(hand_settings, sel, GREEN, brush_size)
            notification = f"Green  →  {get_selected_text(sel)}"
            notif_time = time.time()
            print(notification)

        elif key == ord("4"):
            apply_color_to_selected(hand_settings, sel, YELLOW, brush_size)
            notification = f"Yellow  →  {get_selected_text(sel)}"
            notif_time = time.time()
            print(notification)

        elif key == ord("e"):
            apply_color_to_selected(hand_settings, sel, BLACK, ERASER_THICKNESS)
            notification = f"Eraser  →  {get_selected_text(sel)}"
            notif_time = time.time()
            print(notification)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
