"""Recompute saved group evidence on CPU without inference or run-state changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from controlled_statistics import (regression_checks, source_components,
                                   stat, validate_controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    data = args.input.read_bytes()
    old = json.loads(data)
    records = old['records']
    tests = regression_checks()
    components = source_components(records)
    controls = validate_controls(records)
    primary = [r for r in records if r['score'] == config['analysis']['primary_score']
               and r['nms'] == config['analysis']['primary_nms'] and r['level'] == 0]
    assert all(r['adj']['final']['resolved'] for r in primary)
    cohort = sorted({r['group'] for r in primary})
    assert all(sum(r['group'] == g for r in primary) == 2 for g in cohort)
    assert len(records) == len(cohort) * 2 * 4 * len(config['scores']) * len(config['nms'])
    summary = stat(records, components, config['bootstrap'], config['seed'])
    primary_table = {}
    for detector in ('orcnn', 'retina'):
        primary_table[detector] = [
            summary[f'{detector}:0.25:0.1:{level}'] for level in range(4)]
    candidates = []
    for level in (1, 2, 3):
        both = [primary_table[d][level] for d in ('orcnn', 'retina')]
        if all(x['target_recall']['estimate_pct'] > 50
               and x['effect']['estimate_pp'] >= 10
               and x['effect']['ci95_pp'][0] > 0 for x in both):
            candidates.append(level)
    # This review may reject the prerequisite, but does not grant continue
    # without the separately required event and sensitivity conditions.
    provenance = [a['evidence'] for a in old['acceptance']]
    root = args.config.resolve().parent.parent
    unavailable = [p for p in provenance if not (root / p).is_file()]
    full_ids = {(a['image_id'], tuple(a['members'])) for a in old['acceptance']}
    group_sizes = {r['group']: r['adj']['pre']['gt'] for r in primary}
    assert sum(group_sizes.values()) == controls
    output = {
        'purpose': 'paired_group_reaggregation_of_existing_predictions',
        'source_result_sha256': hashlib.sha256(data).hexdigest(),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'sample_flow': old['sample_flow'],
        'component_groups': components,
        'smd_final_inherited_not_recomputed': old['smd_final'],
        'regression_checks_executed': tests,
        'review_images_missing': unavailable,
        'review_image_existence_is_not_independent_manual_review': True,
        'accepted_identities': len(full_ids),
        'primary_group_sizes': group_sizes,
        'all_accepted_group_supplement_present': len(cohort) == len(full_ids),
        'cohort_clear_passed': True,
        'matched_control_sources_unique': controls,
        'primary_levels_passing_joint_effect_prerequisite': candidates,
        'investment_decision': 'stop_on_inherited_cohort' if not candidates else 'requires_event_and_sensitivity_review',
        'summary': summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({
        'groups': len(cohort), 'source_components': len(components),
        'conditions': len(summary), 'regression': tests,
        'joint_effect_levels': candidates,
        'primary': {d: [{k: row[k] for k in ('effect', 'adj_unresolved', 'ctrl_unresolved',
                     'adj_merge_increment', 'adj_duplicate_increment', 'adj_miss_increment',
                     'target_recall')} for row in primary_table[d]] for d in primary_table},
        'missing_review_images': len(unavailable),
        'full_accepted_supplement_present': output['all_accepted_group_supplement_present'],
        'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == '__main__':
    main()
