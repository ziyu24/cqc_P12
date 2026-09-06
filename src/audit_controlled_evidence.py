#!/usr/bin/env python3
"""Read-only CPU audit of the existing controlled-analysis implementation.

This reproduces implementation defects, not a degradation experiment.  It does
not modify environments, checkpoints, run state, or the analysis under audit.
Run from the project root with its existing inference dependencies available.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import pickle
import types
import warnings
from pathlib import Path

import networkx as nx
import numpy as np


class NumpyMetadataUnpickler(pickle.Unpickler):
    """Read known archived NumPy metadata without global module aliases.

    This is a compatibility adapter, not a security-restricted unpickler.
    Only use it for trusted project checkpoints.
    """

    def find_class(self, module, name):
        if module == "numpy._core.multiarray" and name in ("_reconstruct", "scalar"):
            module = "numpy.core.multiarray"
        return super().find_class(module, name)


def load_archived_checkpoint(path):
    import torch

    compat = types.ModuleType("numpy_checkpoint_compat")
    compat.Unpickler = NumpyMetadataUnpickler
    compat.load = pickle.load
    compat.loads = pickle.loads
    return torch.load(path, map_location="cpu", pickle_module=compat)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary_probe(source):
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("dependency_components", "summary")]
    namespace = {"np": np, "nx": nx}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    records = []
    for level in range(4):
        records.append({
            "record_key": f"orcnn:0:{level}:0.25:0.1", "group_id": 0,
            "adjacent_image": "a", "control_images": ["b", "c"],
            "detector": "orcnn", "score": .25, "nms": .1, "level": level,
            "adjacent": {"resolved": level == 0, "merge": False,
                         "duplicate": False, "miss": level > 0},
            "control_resolved": True,
        })
    # The actual production summarizer must yield 100pp at every degraded level.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result, components = namespace["summary"](records, 1, 20)
    observed = result["orcnn"]["3"]["effect"]["estimate_pp"]
    return {"expected_effect_pp": 100.0,
            "observed_effect_pp": observed if math.isfinite(observed) else None,
            "observed_nonfinite": not math.isfinite(observed),
            "graph_components": len(components),
            "summary_components": result["orcnn"]["3"]["effect"]["dependency_components"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--load-observer", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    population_path = Path(config["population"])
    population = json.loads(population_path.read_text())
    groups = population["audit_order"][:config["accepted_group_count"]]
    adjacent_images = {group["image_id"] for group in groups}
    isolated = [item for item in population["isolated_candidates"]
                if item["image_id"] not in adjacent_images]

    from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
    from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

    result = {
        "audit_kind": "implementation_and_checkpoint_only_no_effect_estimate",
        "source_sha256": sha256(args.source),
        "config_sha256": sha256(args.config),
        "population_sha256": sha256(population_path),
        "sample_capacity": {
            "adjacent_groups": len(groups), "adjacent_images": len(adjacent_images),
            "control_demand": sum(len(group["members"]) for group in groups),
            "isolated_targets": len(isolated),
            "isolated_source_images": len({item["image_id"] for item in isolated}),
            "covariate_common_support": "not_evaluated",
        },
        "production_summary_probe": summary_probe(args.source),
        "roi_has_bbox_post_process": hasattr(Shared2FCBBoxHead, "_bbox_post_process"),
        "dense_post_process_contains_rescale": "scale_boxes" in inspect.getsource(
            BaseDenseHead._bbox_post_process),
    }
    if args.load_observer:
        import torch
        from mmdet.apis import init_detector

        observer = config["orcnn"]
        checkpoint = load_archived_checkpoint(observer["checkpoint"])
        state = checkpoint["state_dict"]
        model = init_detector(observer["config"], checkpoint=None, device="cpu")
        model.load_state_dict(state, strict=True)
        model.dataset_meta = checkpoint.get("meta", {}).get("dataset_meta", {})
        result["archived_observer"] = {
            "tensor_keys": len(state),
            "all_values_are_tensors": all(isinstance(value, torch.Tensor) for value in state.values()),
            "all_values_finite": all(torch.isfinite(value).all().item() for value in state.values()),
            "strict_model_load": True,
            "checkpoint_sha256": sha256(observer["checkpoint"]),
            "observer_config_sha256": sha256(observer["config"]),
            "cpu_forward_tested": False, "gpu_forward_tested": False,
            "clear_ap50": None,
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
