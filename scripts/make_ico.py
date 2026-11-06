#!/usr/bin/env python3
"""PNG -> Windows .ico with the usual sizes.   python3 scripts/make_ico.py icon.png matita.ico

Uses Pillow when it's installed; otherwise writes a single 256-pixel PNG entry (Windows scales it)."""
import struct
import sys

src, dst = sys.argv[1], sys.argv[2]
try:
    from PIL import Image
    Image.open(src).convert("RGBA").save(dst, sizes=[(16, 16), (20, 20), (24, 24), (32, 32), (40, 40),
                                                      (48, 48), (64, 64), (128, 128), (256, 256)])
except ImportError:
    png = open(src, "rb").read()
    w, h = struct.unpack(">II", png[16:24])
    entry = struct.pack("<BBBBHHII", w % 256, h % 256, 0, 0, 1, 32, len(png), 6 + 16)
    with open(dst, "wb") as fh:
        fh.write(struct.pack("<HHH", 0, 1, 1) + entry + png)
