"""Group-level paired statistics for the controlled adjacency experiment."""
from __future__ import annotations

import copy
import numpy as np


def source_components(records):
    """One node per group; any shared source image joins its groups."""
    images = {}
    for r in records:
        identity = (r['image'], tuple(r['controls']))
        if r['group'] in images and images[r['group']] != identity:
            raise ValueError('A group changes its source images across conditions')
        images[r['group']] = identity
    parent = {g: g for g in images}

    def root(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    owners = {}
    for g, (adj, controls) in sorted(images.items()):
        for image in (adj, *controls):
            if image in owners:
                parent[root(g)] = root(owners[image])
            owners[image] = g
    groups = {}
    for g in sorted(images):
        groups.setdefault(root(g), []).append(g)
    return sorted(groups.values(), key=lambda x: x[0])


def validate_controls(records):
    identities = {r['group']: (r['image'], tuple(r['controls'])) for r in records}
    adjacent = {v[0] for v in identities.values()}
    controls = [i for _, sources in identities.values() for i in sources]
    if len(controls) != len(set(controls)) or adjacent.intersection(controls):
        raise ValueError('Control source images must be unique and disjoint')
    return len(controls)


def group_outcomes(r):
    """The control outcome is failure of the WHOLE size-matched pseudogroup."""
    adj = r['adj']['final']
    controls = [x['final'] for x in r['ctrl']]
    result = {
        'adj_unresolved': float(not adj['resolved']),
        'ctrl_unresolved': float(not all(x['resolved'] for x in controls)),
    }
    for event in ('merge', 'duplicate', 'miss', 'cardinality_error'):
        result['adj_' + event] = float(adj[event])
        result['ctrl_' + event] = float(any(x[event] for x in controls))
    result['adj_count_error'] = len(adj['candidate_ids']) - r['adj']['pre']['gt']
    result['adj_cardinality_mae'] = abs(result['adj_count_error'])
    return result


def paired_values(r, clear):
    value, baseline = group_outcomes(r), group_outcomes(clear)
    value['effect'] = (value['adj_unresolved'] - baseline['adj_unresolved']
                       - value['ctrl_unresolved'] + baseline['ctrl_unresolved'])
    for event in ('merge', 'duplicate', 'miss', 'cardinality_error'):
        a, c = 'adj_' + event, 'ctrl_' + event
        value[a + '_increment'] = value[a] - baseline[a]
        value[c + '_increment'] = value[c] - baseline[c]
        value[event + '_difference_in_differences'] = (
            value[a + '_increment'] - value[c + '_increment'])
    return value


def _counts(r, side, stage):
    observations = [r['adj']] if side == 'adj' else r['ctrl']
    denominator = sum(x['pre']['gt'] for x in observations)
    if stage == 'final':
        numerator = sum(len(x['final']['matching']) for x in observations)
    else:
        numerator = sum(x[stage]['matched'] for x in observations)
    return numerator, denominator


def stat(records, comps, reps, seed):
    """Fixed cohort, paired cluster bootstrap, group-weighted estimands.

    comps defines the primary cohort; outside groups are excluded at this real
    summarization boundary. Every included condition must have all its groups.
    """
    comps = sorted([sorted(x) for x in comps], key=lambda x: x[0])
    groups = sorted(g for comp in comps for g in comp)
    if not groups or len(groups) != len(set(groups)):
        raise ValueError('Components must be a nonempty partition of groups')
    index = {g: i for i, g in enumerate(groups)}
    membership = np.zeros((len(comps), len(groups)))
    for j, comp in enumerate(comps):
        membership[j, [index[g] for g in comp]] = 1
    draws = np.random.default_rng(seed).integers(0, len(comps), (reps, len(comps)))
    multiplicity = np.zeros((reps, len(comps)))
    np.add.at(multiplicity, (np.arange(reps)[:, None], draws), 1)
    weights = multiplicity @ membership
    denominator = weights.sum(1)
    conditions = {}
    for r in records:
        if r['group'] not in index:
            continue
        key = (r['detector'], r['score'], r['nms'], r['level'])
        bucket = conditions.setdefault(key, {})
        if r['group'] in bucket:
            raise ValueError('Duplicate group/condition record')
        bucket[r['group']] = r
    if not conditions or any(set(b) != set(groups) for b in conditions.values()):
        raise ValueError('A condition is missing fixed-cohort groups')

    def interval(value, samples, unit='pp'):
        scale = 1 if unit == 'count' else 100
        suffix = '' if unit == 'count' else '_' + unit
        return {'estimate' + suffix: round(float(value * scale), 6),
                'ci95' + suffix: [round(float(q * scale), 6)
                                 for q in np.quantile(samples, [.025, .975])]}

    output = {}
    for key, rows in sorted(conditions.items()):
        clear = conditions[key[:3] + (0,)]
        values = [paired_values(rows[g], clear[g]) for g in groups]
        names = list(values[0])
        matrix = np.asarray([[v[n] for n in names] for v in values])
        means = matrix.mean(0)
        samples = (weights @ matrix) / denominator[:, None]
        result = {name: interval(means[j], samples[:, j],
                  'count' if name in ('adj_count_error', 'adj_cardinality_mae') else 'pp')
                  for j, name in enumerate(names)}
        for side in ('adj', 'ctrl'):
            coverage = {}
            for stage in ('pre', 'post_untruncated', 'final'):
                counts = np.asarray([_counts(rows[g], side, stage) for g in groups])
                num, den = counts[:, 0], counts[:, 1]
                bs = (weights @ num) / (weights @ den)
                coverage[stage] = interval(num.sum() / den.sum(), bs, 'pct')
                coverage[stage].update(matched=int(num.sum()), gt=int(den.sum()))
                if side == 'ctrl' and stage == 'final':
                    result['target_recall'] = dict(coverage[stage])
                    result['target_recall']['group_mean_pct'] = float(100 * np.mean(num / den))
            for label, a, b in [('nms_loss', 'pre', 'post_untruncated'),
                                 ('truncation_loss', 'post_untruncated', 'final')]:
                first = np.asarray([_counts(rows[g], side, a) for g in groups])
                second = np.asarray([_counts(rows[g], side, b) for g in groups])
                lost, den = first[:, 0] - second[:, 0], first[:, 1]
                coverage[label] = interval(lost.sum() / den.sum(),
                                          (weights @ lost) / (weights @ den), 'pp')
                coverage[label]['matched_count_loss'] = int(lost.sum())
            result[side + '_coverage'] = coverage
        output[':'.join(map(str, key))] = result
    return output


def regression_checks():
    """Counterexamples exercise the same functions used by production."""
    def observation(resolved, n=1):
        return {'pre': {'gt': n, 'matched': n},
                'post_untruncated': {'matched': n if resolved else n - 1},
                'final': {'resolved': resolved, 'merge': False, 'duplicate': False,
                          'miss': not resolved, 'cardinality_error': not resolved,
                          'matching': list(range(n if resolved else n - 1)),
                          'candidate_ids': list(range(n if resolved else n - 1))}}

    def record(group, level, failed, partial=False):
        return {'group': group, 'image': 'a' + str(group),
                'controls': ['b' + str(group), 'c' + str(group)],
                'detector': 'orcnn', 'score': .25, 'nms': .1, 'level': level,
                'adj': observation(not failed, 2),
                'ctrl': [observation(not partial), observation(True)]}

    rows = [record(0, level, level > 0) for level in range(4)]
    result = stat(rows, [[0]], 40, 1)
    assert all(result[f'orcnn:0.25:0.1:{l}']['effect']['estimate_pp'] == 100
               for l in (1, 2, 3))
    partial = [record(0, 0, False), record(0, 1, True, partial=True)]
    assert stat(partial, [[0]], 40, 1)['orcnn:0.25:0.1:1']['effect']['estimate_pp'] == 0
    uneven = [record(g, l, l > 0 and g < 2) for g in range(3) for l in range(2)]
    assert abs(stat(uneven, [[0, 1], [2]], 40, 1)['orcnn:0.25:0.1:1']
               ['effect']['estimate_pp'] - 200 / 3) < 1e-5
    outside = rows + [record(8, l, False) for l in range(4)]
    assert stat(outside, [[0]], 40, 1) == result
    shared = [copy.deepcopy(record(g, 0, False)) for g in (0, 1)]
    shared[1]['controls'][0] = shared[0]['controls'][0]
    assert source_components(shared) == [[0, 1]]
    try:
        validate_controls(shared)
    except ValueError:
        pass
    else:
        raise AssertionError('Shared control images were not rejected')
    return {'one_sided_four_level_effect_100pp': True,
            'partial_control_failure_is_whole_pseudogroup_failure': True,
            'unequal_components_weighted_by_group': True,
            'outside_cohort_cannot_change_summary': True,
            'shared_control_joins_components_and_is_rejected': True}
