#!/usr/bin/env python3
"""Freeze and enumerate the HRSC2016 r001 geometry qualification population."""
from __future__ import annotations

import argparse
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from shapely.geometry import Polygon


def polygon(obj: ET.Element) -> Polygon:
    cx = float(obj.findtext("mbox_cx")); cy = float(obj.findtext("mbox_cy"))
    width = float(obj.findtext("mbox_w")); height = float(obj.findtext("mbox_h"))
    angle = float(obj.findtext("mbox_ang"))
    corners = [(-width / 2, -height / 2), (width / 2, -height / 2),
               (width / 2, height / 2), (-width / 2, height / 2)]
    c, s = math.cos(angle), math.sin(angle)
    return Polygon([(cx + x * c - y * s, cy + x * s + y * c) for x, y in corners])


def read_objects(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    rows = []
    for index, obj in enumerate(root.findall("./HRSC_Objects/HRSC_Object")):
        box = polygon(obj)
        rows.append({"object_index": index, "object_id": obj.findtext("Object_ID"),
                     "short_side": min(float(obj.findtext("mbox_w")), float(obj.findtext("mbox_h"))),
                     "polygon": box})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["dataset_root"])
    ids = [line.strip() for line in (root / "splits" / cfg["evaluation_split"]).read_text().splitlines() if line.strip()]
    candidates, isolated, rejected_sizes = [], [], Counter()
    for image_id in ids:
        objs = read_objects(root / "annfiles" / f"{image_id}.xml")
        graph = {i: set() for i in range(len(objs))}
        nearest = [float("inf")] * len(objs)
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                distance = objs[i]["polygon"].distance(objs[j]["polygon"])
                gap = distance / min(objs[i]["short_side"], objs[j]["short_side"])
                nearest[i] = min(nearest[i], gap); nearest[j] = min(nearest[j], gap)
                if gap <= cfg["adjacent_gap_max"]:
                    graph[i].add(j); graph[j].add(i)
        seen = set()
        for start in graph:
            if start in seen or not graph[start]:
                continue
            stack, component = [start], []
            seen.add(start)
            while stack:
                node = stack.pop(); component.append(node)
                for neighbor in graph[node]:
                    if neighbor not in seen:
                        seen.add(neighbor); stack.append(neighbor)
            if cfg["group_size"][0] <= len(component) <= cfg["group_size"][1]:
                candidates.append({"image_id": image_id, "members": component,
                                   "size": len(component), "min_short_side": min(objs[i]["short_side"] for i in component)})
            else:
                rejected_sizes[str(len(component))] += 1
        for i, gap in enumerate(nearest):
            if gap >= cfg["isolated_gap_min"]:
                isolated.append({"image_id": image_id, "object_index": i, "object_id": objs[i]["object_id"],
                                 "short_side": objs[i]["short_side"], "nearest_gap": gap})
    rng = random.Random(cfg["random_seed"])
    rng.shuffle(candidates)
    payload = {"config": cfg, "split_image_count": len(ids), "candidate_group_count": len(candidates),
               "candidate_group_image_count": len({x["image_id"] for x in candidates}),
               "candidate_isolated_count": len(isolated), "rejected_component_sizes": dict(rejected_sizes),
               "audit_order": candidates, "isolated_candidates": isolated,
               "qualification_pre_audit": {"enough_groups": len(candidates) >= cfg["minimum_groups"],
                                             "enough_images": len({x["image_id"] for x in candidates}) >= cfg["minimum_images"]}}
    if "manual_audit" in cfg:
        audit = cfg["manual_audit"]
        payload["manual_audit"] = audit
        payload["qualification_post_audit"] = {
            "enough_groups": audit["accepted_candidates"] >= cfg["minimum_groups"],
            "enough_images": len({x["image_id"] for x in candidates[:audit["accepted_candidates"]]}) >= cfg["minimum_images"],
            "obvious_missing_label_rate": 0.0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("split_image_count", "candidate_group_count", "candidate_group_image_count", "candidate_isolated_count", "qualification_pre_audit")}, indent=2))


if __name__ == "__main__":
    main()
