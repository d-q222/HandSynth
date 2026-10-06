"""Draw the app icon (1024×1024 PNG): stacked chord bars, wide bass to narrow top, in the overlay's colours."""
import sys

from PIL import Image, ImageDraw

S = 1024
out = sys.argv[1] if len(sys.argv) > 1 else "icon.png"

img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
bg = Image.new("RGBA", (S, S))
top, bottom = (40, 44, 52), (16, 18, 22)
for y in range(S):                                   # vertical gradient
    t = y / (S - 1)
    ImageDraw.Draw(bg).line([(0, y), (S, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
mask = Image.new("L", (S, S), 0)
inset = 100                                          # macOS icon grid: artwork inset from the canvas edge
ImageDraw.Draw(mask).rounded_rectangle([inset, inset, S - inset, S - inset], radius=185, fill=255)
img.paste(bg, (0, 0), mask)

d = ImageDraw.Draw(img)
bars = [  # (width, colour) bottom → top: bass wide, upper voices narrower, like the balanced voicing
    (560, (94, 234, 212)),     # teal: chord / bass
    (420, (251, 191, 36)),     # amber: expression
    (300, (245, 245, 245)),
    (190, (245, 245, 245)),
]
h, gap = 70, 44
y = S // 2 + (len(bars) * (h + gap) - gap) // 2 - h
for w, colour in bars:
    x0 = (S - w) // 2
    d.rounded_rectangle([x0, y, x0 + w, y + h], radius=h // 2, fill=colour + (255,))
    y -= h + gap
img.save(out)
