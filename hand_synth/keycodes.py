"""Arrow-key decoding for cv2.waitKeyEx across macOS, Windows and Linux (GTK)."""
_ARROWS = {
    63234: "left", 63235: "right", 63232: "up", 63233: "down",        # macOS
    2424832: "left", 2555904: "right", 2490368: "up", 2621440: "down",  # Windows
    65361: "left", 65363: "right", 65362: "up", 65364: "down",        # Linux GTK
}


def arrow(k):
    return _ARROWS.get(k)
