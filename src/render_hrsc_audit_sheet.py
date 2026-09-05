#!/usr/bin/env python3
"""Render frozen HRSC candidate groups for blind annotation-completeness review."""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw


def corners(obj: ET.Element) -> list[tuple[float, float]]:
    cx, cy = float(obj.findtext("mbox_cx")), float(obj.findtext("mbox_cy"))
    w, h, angle = float(obj.findtext("mbox_w")), float(obj.findtext("mbox_h")), float(obj.findtext("mbox_ang"))
    c, s = math.cos(angle), math.sin(angle)
    return [(cx + x*c-y*s, cy+x*s+y*c) for x, y in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    args = ap.parse_args()
    rows = json.loads(args.population.read_text())["audit_order"][args.start:args.start + args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    tiles: list[tuple[int, Image.Image]] = []
    for rank, row in enumerate(rows, args.start + 1):
        image_id = row["image_id"]
        image = Image.open(args.dataset_root / "images" / f"{image_id}.bmp").convert("RGB")
        objects = ET.parse(args.dataset_root / "annfiles" / f"{image_id}.xml").getroot().findall("./HRSC_Objects/HRSC_Object")
        draw = ImageDraw.Draw(image)
        for index in row["members"]:
            draw.line(corners(objects[index]) + [corners(objects[index])[0]], fill=(0, 255, 0), width=4)
        image.thumbnail((500, 350))
        image.save(args.output / f"{rank:03d}_{image_id}_k{row['size']}.jpg", quality=92)
        tiles.append((rank, image.copy()))
    for offset in range(0, len(tiles), 10):
        batch = tiles[offset:offset + 10]
        sheet = Image.new("RGB", (1000, 1850), "white")
        label = ImageDraw.Draw(sheet)
        for position, (rank, tile) in enumerate(batch):
            x, y = (position % 2) * 500, (position // 2) * 370
            sheet.paste(tile, (x, y + 20))
            label.text((x + 4, y + 2), str(rank), fill="black")
        sheet.save(args.output / f"sheet_{args.start + offset + 1:03d}_{args.start + offset + len(batch):03d}.jpg", quality=90)


if __name__ == "__main__":
    main()
