"""Next-batch trial planning.

The planner answers one question: given only the records visible at a chosen
time, which manufacturable combinations are worth spending this week's trials
on, and why?  It deliberately does *not* collapse strength and loss into a
single utility.  Selection uses Pareto dominance, declared performance floors
and information gain, never a weighted composite score.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from .common import finite, require
from .domain import validate
from .model import ModelError, fit_pair

ENGINE_VERSION = '1.0.0'
MIN_COMPLETED = 6
MIN_BATCHES = 3
MAX_CANDIDATES = 6000
REQUEST_KEYS = {
    'at', 'n_trials', 'cost_budget', 'strength_floor', 'loss_ceiling',
    'component_bounds', 'confidence', 'intent', 'min_separation',
    'exploration_distance', 'extrapolation_margin', 'seed',
    'repeat_tolerance_strength', 'repeat_tolerance_loss',
}


class PlanningError(ValueError):
    pass


def _erfinv(value):
    from statistics import NormalDist
    return NormalDist().inv_cdf((value + 1.0) / 2.0)


def normalize(data, request):
    """Validate a planning request and fill in data-independent defaults."""
    validate(data)
    if not isinstance(request, dict):
        raise PlanningError('规划请求必须是对象')
    unknown = set(request) - REQUEST_KEYS
    if unknown:
        raise PlanningError(f'未知的规划参数: {", ".join(sorted(unknown))}')
    constraints = data['constraints']
    total = float(constraints['total_fraction'])
    components = list(constraints['components'])
    notes = []

    at = request.get('at')
    if at is not None:
        require(type(at) is int and at >= 0, 'at 必须是非负整数')
    n_trials = request.get('n_trials', 1)
    require(type(n_trials) is int and 1 <= n_trials <= 20, 'n_trials 必须是 1 到 20 的整数')

    max_cost = float(constraints['max_cost'])
    budget = request.get('cost_budget')
    if budget is None:
        budget = max_cost
    else:
        finite(budget, 'cost_budget', 0)
        if budget > max_cost:
            notes.append(f'本周成本上限 {budget:g} 高于资料声明的 max_cost {max_cost:g}，按 {max_cost:g} 处理。')
            budget = max_cost

    strength_floor = request.get('strength_floor')
    if strength_floor is not None:
        finite(strength_floor, 'strength_floor')
    loss_ceiling = request.get('loss_ceiling')
    if loss_ceiling is not None:
        finite(loss_ceiling, 'loss_ceiling', 0)

    confidence = request.get('confidence', 0.8)
    finite(confidence, 'confidence')
    require(0.5 <= confidence <= 0.999, 'confidence 必须在 0.5 到 0.999 之间')

    intent = request.get('intent', 'auto')
    require(intent in ('auto', 'confirm', 'explore'), 'intent 必须是 auto、confirm 或 explore')

    min_separation = request.get('min_separation', 0.04)
    finite(min_separation, 'min_separation', 0)
    require(min_separation <= 0.5, 'min_separation 过大')
    exploration_distance = request.get('exploration_distance', 0.06)
    finite(exploration_distance, 'exploration_distance', 0)
    require(exploration_distance <= 1.0, 'exploration_distance 过大')
    extrapolation_margin = request.get('extrapolation_margin', 0.02)
    finite(extrapolation_margin, 'extrapolation_margin', 0)
    require(extrapolation_margin <= 0.1, 'extrapolation_margin 过大')

    repeat_strength = request.get('repeat_tolerance_strength')
    if repeat_strength is not None:
        finite(repeat_strength, 'repeat_tolerance_strength', 0)
    repeat_loss = request.get('repeat_tolerance_loss')
    if repeat_loss is not None:
        finite(repeat_loss, 'repeat_tolerance_loss', 0)

    seed = request.get('seed', 0)
    require(type(seed) is int, 'seed 必须是整数')

    bounds = {}
    supplied = request.get('component_bounds') or {}
    require(isinstance(supplied, dict), 'component_bounds 必须是对象')
    unknown_components = set(supplied) - set(components)
    if unknown_components:
        raise PlanningError(f'component_bounds 含未知组分: {", ".join(sorted(unknown_components))}')
    for name in components:
        pair = supplied.get(name, [0.0, total])
        require(isinstance(pair, (list, tuple)) and len(pair) == 2, 'component_bounds 的每个组分需要 [下限, 上限]')
        low = finite(pair[0], f'component_bounds.{name} 下限', 0)
        high = finite(pair[1], f'component_bounds.{name} 上限', 0)
        require(low <= high, f'component_bounds.{name} 下限不能大于上限')
        require(high <= total, f'component_bounds.{name} 上限不能超过 total_fraction')
        bounds[name] = [float(low), float(high)]

    return {
        'at': at,
        'n_trials': n_trials,
        'cost_budget': float(budget),
        'strength_floor': None if strength_floor is None else float(strength_floor),
        'loss_ceiling': None if loss_ceiling is None else float(loss_ceiling),
        'component_bounds': bounds,
        'confidence': float(confidence),
        'intent': intent,
        'min_separation': float(min_separation),
        'exploration_distance': float(exploration_distance),
        'extrapolation_margin': float(extrapolation_margin),
        'repeat_tolerance_strength': None if repeat_strength is None else float(repeat_strength),
        'repeat_tolerance_loss': None if repeat_loss is None else float(repeat_loss),
        'seed': seed,
        'notes': notes,
    }


def suggest(data, request):
    """Normalize a request, run the plan, and attach the normalized request."""
    normalized = normalize(data, request)
    notes = list(normalized.pop('notes'))
    result = plan(data, normalized, notes)
    result['request'] = normalized
    return result


def _front_mask(matrix):
    """Boolean mask of rows not dominated by another row (all columns maximized)."""
    keep = np.ones(len(matrix), dtype=bool)
    for index in range(len(matrix)):
        if not keep[index]:
            continue
        dominated = np.all(matrix >= matrix[index], axis=1) & np.any(matrix > matrix[index], axis=1)
        if np.any(dominated):
            keep[index] = False
    return keep


def _compositions(units, parts):
    if parts == 1:
        yield (units,)
        return
    for used in range(units + 1):
        for rest in _compositions(units - used, parts - 1):
            yield (used,) + rest


def _candidates(components, bounds, total, seed):
    dimension = len(components)
    if dimension < 2:
        raise PlanningError('至少需要两个组分才能形成配方空间')
    points = []
    if dimension <= 3:
        # Choose the finest grid whose simplex capacity stays within the cap,
        # rather than the first coarse grid that fits.
        for step in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2):
            units = int(round(total / step))
            if units <= 0:
                continue
            if math.comb(units + dimension - 1, dimension - 1) <= MAX_CANDIDATES:
                points = [np.array(row, dtype=float) * (total / units) for row in _compositions(units, dimension)]
                break
    if not points:
        rng = np.random.default_rng(seed)
        draws = rng.dirichlet(np.ones(dimension), size=MAX_CANDIDATES)
        points = [row * total for row in draws]
    return np.asarray(points, dtype=float)


def _box_distances(points, lowest, highest):
    """Euclidean distance from each point to the observed component box."""
    outside = np.maximum(np.maximum(lowest[None, :] - points, points - highest[None, :]), 0.0)
    return np.sqrt(np.sum(outside * outside, axis=1))


def _pairwise_min_distance(points, references):
    if len(references) == 0:
        return np.full(len(points), np.inf)
    diff = points[:, None, :] - references[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)


def _pairwise_distance(a, b):
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)))


def _nearest_neighbour(points):
    """Distance from each row to its nearest *other* row."""
    if len(points) < 3:
        return np.full(len(points), np.inf)
    diff = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(distance, np.inf)
    return distance.min(axis=1)


def _observed_frontier(completed, components):
    rows = []
    for trial in completed:
        dominated = any(
            other['strength'] >= trial['strength'] and other['loss'] <= trial['loss']
            and (other['strength'] > trial['strength'] or other['loss'] < trial['loss'])
            for other in completed
        )
        if not dominated:
            rows.append({
                'id': trial['id'],
                'batch': trial['batch'],
                'fractions': {name: float(trial['fractions'][name]) for name in components},
                'strength': float(trial['strength']),
                'loss': float(trial['loss']),
            })
    rows.sort(key=lambda row: (-row['strength'], row['loss']))
    return rows


def _frontier_probabilities(candidates, frontier, tolerance):
    """Probability a new trial is no worse than the observed frontier.

    Observed values contain measurement and batch noise, so comparing a shrunk
    posterior mean directly with the best observed value systematically favours
    repeating the current leader.  Each candidate is instead compared with the
    whole posterior predictive distribution of a new furnace, allowing an
    explicit ``tolerance`` on both responses.  Strength and loss are evaluated
    separately; no weight between them is introduced.
    """
    if not frontier:
        for candidate in candidates:
            candidate['frontier_probability'] = None
            candidate['frontier_reference'] = None
            return
    strengths = np.array([row['strength'] for row in frontier])
    losses = np.array([row['loss'] for row in frontier])
    for candidate in candidates:
        mean_strength = candidate['predicted']['strength']['mean']
        std_strength = max(candidate['predicted']['strength']['std'], 1e-12)
        mean_loss = candidate['predicted']['loss']['mean']
        std_loss = max(candidate['predicted']['loss']['std'], 1e-12)
        p_strength = ndtr((mean_strength - (strengths - tolerance['strength'])) / std_strength)
        p_loss = ndtr(((losses + tolerance['loss']) - mean_loss) / std_loss)
        joint = p_strength * p_loss
        best = int(np.argmax(joint))
        candidate['frontier_probability'] = float(joint[best])
        candidate['frontier_reference'] = frontier[best]['id']


def _payload(candidate, components, confidence):
    z = float(np.sqrt(2.0) * _erfinv(2.0 * confidence - 1.0))
    strength = candidate['predicted']['strength']
    loss = candidate['predicted']['loss']
    return {
        'fractions': {name: float(candidate['fractions'][i]) for i, name in enumerate(components)},
        'cost': float(candidate['cost']),
        'role': candidate['role'],
        'predicted': {
            'strength': {
                'mean': strength['mean'],
                'std': strength['std'],
                'interval': [strength['mean'] - z * strength['std'], strength['mean'] + z * strength['std']],
            },
            'loss': {
                'mean': loss['mean'],
                'std': loss['std'],
                'interval': [max(0.0, loss['mean'] - z * loss['std']), loss['mean'] + z * loss['std']],
            },
        },
        'floor_probabilities': candidate['floor_probabilities'],
        'information_gain': candidate['information_gain'],
        'support': candidate['support'],
        'rationale': None,
    }


def plan(data, request, initial_notes=None):
    validate(data)
    constraints = data['constraints']
    components = list(constraints['components'])
    costs = {name: float(value) for name, value in constraints['costs'].items()}
    total = float(constraints['total_fraction'])
    notes = list(initial_notes or [])
    reasons = []
    request = dict(request)

    cutoff = request['at']
    if cutoff is None:
        cutoff = max((trial['visible_at'] for trial in data['trials']), default=0)
    visible = [trial for trial in data['trials'] if trial['visible_at'] <= cutoff]
    completed = [trial for trial in visible if trial['status'] == 'complete']
    failed = [trial for trial in visible if trial['status'] == 'failed']
    pending = [trial for trial in visible if trial['status'] == 'pending']
    later = [trial for trial in data['trials'] if trial['visible_at'] > cutoff]

    basis = {
        'visible_at': cutoff,
        'completed_trials': len(completed),
        'distinct_batches': len({trial['batch'] for trial in completed}),
        'failed_trials': len(failed),
        'pending_trials': len(pending),
        'records_after_cutoff': len(later),
        'completed_ids': [trial['id'] for trial in completed],
        'failed_ids': [trial['id'] for trial in failed],
        'pending_ids': [trial['id'] for trial in pending],
    }

    state = {'model': None, 'support_rule': None, 'budget': None}

    def finish(status, suggestions, alternatives, extra_reasons):
        result = {
            'status': status,
            'data_basis': basis,
            'model': state['model'],
            'support_rule': state['support_rule'],
            'budget': state['budget'],
            'observed_frontier': _observed_frontier(completed, components) if completed else [],
            'suggestions': suggestions,
            'alternatives': alternatives,
            'reasons': notes + reasons + extra_reasons,
            'meaning': (
                '基于指定时点可见资料的序贯试制建议，不是最优配方证明、不是安全认证，'
                '也不把强度与损耗换算为同一分数。'
            ),
            'engine': {'name': 'grouped-gp-pareto', 'version': ENGINE_VERSION},
        }
        return result

    if later:
        reasons.append(f'只使用 visible_at <= {cutoff} 的记录；另有 {len(later)} 条记录晚于该时点，未参与本次建议。')
    if pending:
        reasons.append(
            f'{len(pending)} 条试验仍在进行，未作为结果参与拟合；其配方及邻近区域也不作为本次建议。'
        )
    if failed:
        reasons.append(
            f'{len(failed)} 条失败试验只作为该区域不可行的证据，没有按强度为零或完整结果处理。'
        )
    if completed:
        reasons.append(
            f'{len(completed)} 条完整记录来自 {basis["distinct_batches"]} 个炉次；'
            '同一炉次的共同误差已计入预测区间，未按独立重复计数。'
        )
    if not failed:
        reasons.append('资料中没有失败试验，安全边界缺少直接证据；建议保持在已完成试验覆盖的组分范围内。')

    if len(completed) < MIN_COMPLETED or basis['distinct_batches'] < MIN_BATCHES:
        return finish('no_recommendation', [], [], [
            f'可用完整观测不足（完成 {len(completed)} 次、独立炉次 {basis["distinct_batches"]} 个；'
            f'至少需要 {MIN_COMPLETED} 次且 {MIN_BATCHES} 个炉次）。'
            '此时给出“最佳配方”会掩盖证据不足，建议先积累完成试验或复核资料。'
        ])

    try:
        models = fit_pair(
            np.array([[trial['fractions'][name] for name in components] for trial in completed]),
            np.array([trial['strength'] for trial in completed], dtype=float),
            np.array([trial['loss'] for trial in completed], dtype=float),
            np.array([trial['batch'] for trial in completed]),
        )
    except (ModelError, np.linalg.LinAlgError, ValueError) as error:
        return finish('no_recommendation', [], [], [f'响应模型无法稳定拟合：{error}。建议补充或复核观测后再规划。'])
    state['model'] = {name: model.diagnostics() for name, model in models.items()}

    points = _candidates(components, request['component_bounds'], total, request['seed'])
    inside_bounds = np.all(
        points >= np.array([request['component_bounds'][name][0] for name in components])[None, :] - 1e-12,
        axis=1,
    ) & np.all(
        points <= np.array([request['component_bounds'][name][1] for name in components])[None, :] + 1e-12,
        axis=1,
    )
    # The upper bounds also apply to the already observed design, otherwise a
    # constraint such as "component b stays below 0.12" would let unreachable
    # high-b observations define where the evidence is said to lie.
    bounds_high = np.array([request['component_bounds'][name][1] for name in components])
    points = points[inside_bounds]
    if len(points) == 0:
        return finish('no_recommendation', [], [], ['制造可行规则与组分守恒没有交集，无法生成候选组合。'])
    costs_array = points @ np.array([costs[name] for name in components])
    affordable = costs_array <= request['cost_budget'] + 1e-12
    if not np.any(affordable):
        return finish('no_recommendation', [], [], [
            f'本周成本上限 {request["cost_budget"]:g} 低于任何可制造组合的成本，无法安排试制。'
        ])
    points = points[affordable]
    costs_array = costs_array[affordable]

    known = [trial['fractions'] for trial in completed + failed + pending]

    def exact_known(point):
        return any(all(abs(point[i] - trial[name]) <= 1e-9 for i, name in enumerate(components)) for trial in known)

    keep = np.array([not exact_known(point) for point in points])
    points = points[keep]
    costs_array = costs_array[keep]
    if len(points) == 0:
        return finish('no_recommendation', [], [], ['可行候选都与已记录配方重复，没有可提供新信息的组合。'])

    completed_points = np.array([[trial['fractions'][name] for name in components] for trial in completed])
    failed_points = np.array([[trial['fractions'][name] for name in components] for trial in failed])
    pending_points = np.array([[trial['fractions'][name] for name in components] for trial in pending])
    # The observed design used for the support envelope is itself restricted to
    # the declared manufacturable bounds.  A directional rule that forbids high
    # component values must not leave unreachable observed rows defining where
    # the evidence is said to lie.
    design = completed_points[np.all(completed_points <= bounds_high + 1e-12, axis=1)]
    if len(design) < MIN_COMPLETED or len({trial['batch'] for trial in completed
                                           if all(trial['fractions'][name] <= bounds_high[i] + 1e-12
                                                  for i, name in enumerate(components))}) < MIN_BATCHES:
        design = completed_points
        notes.append(
            '满足给定制造边界的已完成试验不足，覆盖与安全间隔仍按全部已完成试验计算；'
            '这会使边界附近的建议更保守。'
        )

    nn_completed = _pairwise_min_distance(points, design)
    nn_failed = _pairwise_min_distance(points, failed_points)
    nn_pending = _pairwise_min_distance(points, pending_points)

    # Local support follows the actual design resolution of the experiment book:
    # a candidate counts as supported when it lies within one and a half typical
    # neighbour spacings of a completed formula.
    completed_nn = _nearest_neighbour(design)
    spacings = completed_nn[np.isfinite(completed_nn)]
    q90_nn = float(np.quantile(spacings, 0.9)) if len(spacings) else 0.1
    support_radius = float(np.clip(1.5 * q90_nn, 0.05, 0.12))
    failure_clearance = max(0.05, 0.6 * support_radius)
    pending_clearance = max(0.02, request['min_separation'])

    lowest = design.min(axis=0)
    highest = design.max(axis=0)
    margin = request['extrapolation_margin']
    inside_strict = np.all(points >= lowest[None, :] - 1e-9, axis=1) & np.all(points <= highest[None, :] + 1e-9, axis=1)
    # Local support is a distinct notion from staying inside the observed
    # component ranges.  A grid point on the box boundary can still be a real
    # interpolation if several completed trials lie near it, so support uses
    # the actual nearest completed formula rather than the bounding box, with
    # an optional explicit extrapolation margin outside that box.
    box_distance = _box_distances(points, lowest, highest)
    inside_extended = box_distance <= margin + 1e-12
    supported = inside_extended & (nn_completed <= support_radius + 1e-12)

    # Feasibility envelope.  Compositions satisfy a fixed sum, so they live on
    # a lower-dimensional simplex and a full-dimensional hull cannot be built
    # directly.  The last component is dropped; the remaining coordinates are
    # in one-to-one correspondence with the composition.  A point can sit inside
    # the component bounding box and still lie in a corner no plausible mixture
    # reaches, so the padded hull is used instead of the box alone.
    from scipy.spatial import ConvexHull, QhullError
    reduced = design[:, :-1]
    try:
        hull = ConvexHull(reduced)
        # Signed distance of each candidate to every hull facet; inside means
        # no more than the declared margin beyond a facet.
        deficit = points[:, :-1] @ hull.equations[:, :-1].T + hull.equations[:, -1][None, :]
        in_hull = np.all(deficit <= margin + 1e-12, axis=1)
        hull_status = 'convex_hull_margin'
    except (QhullError, ValueError):
        in_hull = inside_strict | (box_distance <= margin + 1e-12)
        hull_status = 'bounding_box_fallback'
    envelope = in_hull

    outside_components = []
    for point in points:
        outside_components.append([
            name for i, name in enumerate(components)
            if point[i] < lowest[i] - 1e-9 or point[i] > highest[i] + 1e-9
        ])

    exposure = np.zeros(len(points))
    if len(failed_points):
        scale = max(support_radius, 0.05)
        failed_distance = _pairwise_min_distance(points, failed_points)
        failed_w = np.exp(-0.5 * (failed_distance / scale) ** 2)
        completed_w = np.exp(-0.5 * (nn_completed / scale) ** 2)
        exposure = failed_w / np.maximum(failed_w + completed_w, 1e-12)

    safe = envelope
    if len(failed_points):
        safe &= (nn_failed >= failure_clearance - 1e-12) & (exposure <= 0.5)
    safe &= nn_pending >= pending_clearance - 1e-12

    strength_pred = models['strength'].predict(points)
    loss_pred = models['loss'].predict(points)
    floor_probabilities = []
    for index in range(len(points)):
        entry = {'strength': None, 'loss': None, 'joint': None}
        if request['strength_floor'] is not None:
            entry['strength'] = float(ndtr(
                (strength_pred['mean'][index] - request['strength_floor'])
                / max(strength_pred['std'][index], 1e-12)
            ))
        if request['loss_ceiling'] is not None:
            entry['loss'] = float(ndtr(
                (request['loss_ceiling'] - loss_pred['mean'][index])
                / max(loss_pred['std'][index], 1e-12)
            ))
        values = [value for value in (entry['strength'], entry['loss']) if value is not None]
        entry['joint'] = float(min(values)) if values else None
        floor_probabilities.append(entry)

    state['support_rule'] = {
        'support_radius': support_radius,
        'failure_clearance': failure_clearance,
        'pending_clearance': pending_clearance,
        'extrapolation_margin': margin,
        'hull': hull_status,
        'status': 'empirical',
        'meaning': '按试验簿现有间距推导的经验覆盖与安全间隔，不是安全认证。',
    }

    # Each single best observation, with the response the other one had.  A
    # combined "best strength and best loss" would suggest a formula that
    # never existed.
    strength_leader = max(completed, key=lambda trial: trial['strength'])
    loss_leader = min(completed, key=lambda trial: trial['loss'])
    leaders = {
        'strength': {'id': strength_leader['id'], 'strength': float(strength_leader['strength']),
                     'loss': float(strength_leader['loss'])},
        'loss': {'id': loss_leader['id'], 'strength': float(loss_leader['strength']),
                 'loss': float(loss_leader['loss'])},
    }

    declared = request['strength_floor'] is not None or request['loss_ceiling'] is not None
    candidates = []
    for index in range(len(points)):
        mean_strength = float(strength_pred['mean'][index])
        mean_loss = float(loss_pred['mean'][index])
        entry = floor_probabilities[index]
        floors_ok = True if not declared else bool(entry['joint'] is not None and entry['joint'] >= request['confidence'])
        candidates.append({
            'index': index,
            'fractions': points[index],
            'cost': float(costs_array[index]),
            'predicted': {
                'strength': {'mean': mean_strength, 'std': float(strength_pred['std'][index])},
                'loss': {'mean': mean_loss, 'std': float(loss_pred['std'][index])},
            },
            'information_gain': {
                'strength': float(strength_pred['information_gain'][index]),
                'loss': float(loss_pred['information_gain'][index]),
            },
            'floor_probabilities': entry,
            'support': {
                'inside_observed_ranges': bool(inside_strict[index]),
                'inside_feasibility_envelope': bool(envelope[index]),
                'within_support_region': bool(supported[index]),
                'box_distance': float(box_distance[index]),
                'distance_to_nearest_completed': float(nn_completed[index]),
                'distance_to_nearest_failed': None if not len(failed_points) else float(nn_failed[index]),
                'distance_to_nearest_pending': None if not len(pending_points) else float(nn_pending[index]),
                'failure_exposure': float(exposure[index]),
                'outside_strict_components': outside_components[index],
            },
            'floors_met_with_confidence': floors_ok,
            'safe': bool(safe[index]),
            'supported': bool(supported[index]),
            'frontier_probability': None,
            'frontier_reference': None,
            'role': None,
        })

    eligible = [candidate for candidate in candidates if candidate['supported'] and candidate['safe']]
    if not eligible:
        return finish('no_recommendation', [], [], [
            '没有候选组合同时满足可制造、成本、观测覆盖和安全间隔要求；'
            '现有证据不足以支持下一步试制，建议补充试验或放宽经确认的制造边界后再规划。'
        ])

    declared = request['strength_floor'] is not None or request['loss_ceiling'] is not None
    # A boundary region only earns a slot if its chance of meeting the declared
    # floors reaches the confidence the caller already requires.  This is a
    # feasibility gate, not a trade-off weight.
    plausibility = request['confidence']
    frontier = _observed_frontier(completed, components)
    # The tolerance states how much worse than the observed frontier still
    # counts as reproducing it.  It defaults to the observed spread of each
    # response; the caller can shrink or widen it.
    default_strength = float(np.std([trial['strength'] for trial in completed], ddof=1))
    default_loss = float(np.std([trial['loss'] for trial in completed], ddof=1))
    tolerance = {
        'strength': request['repeat_tolerance_strength']
        if request['repeat_tolerance_strength'] is not None else default_strength,
        'loss': request['repeat_tolerance_loss']
        if request['repeat_tolerance_loss'] is not None else default_loss,
    }
    _frontier_probabilities(eligible, frontier, tolerance)
    # A candidate confirms an existing advantage when a new furnace at that
    # formula has a reasonable chance of reproducing the observed frontier, at
    # a location inside the observed component ranges.
    repeat_confidence = 0.5 * request['confidence']

    # Two separate Pareto fronts.  Performance and information are never mixed
    # into one ordering, because a single ordering would implicitly choose a
    # trade-off weight that the business has not supplied.
    interior = [c for c in eligible if c['support']['inside_observed_ranges']]
    if interior:
        interior_matrix = np.array([
            [c['predicted']['strength']['mean'], -c['predicted']['loss']['mean']]
            for c in interior
        ])
        interior_keep = _front_mask(interior_matrix)
        performance_front = [c for c, keep_row in zip(interior, interior_keep) if keep_row]
    else:
        performance_front = []

    boundary = [
        c for c in eligible
        if c['support']['distance_to_nearest_completed'] >= request['exploration_distance']
    ]
    plausibly_feasible = [
        c for c in boundary
        if not declared or (c['floor_probabilities']['joint'] or 0.0) >= plausibility
    ]
    confirms = [
        c for c in performance_front
        if (c['frontier_probability'] or 0.0) >= repeat_confidence
        and (not declared or c['floors_met_with_confidence'])
    ]
    for candidate in confirms:
        candidate['role'] = 'confirm'
    for candidate in plausibly_feasible:
        candidate['role'] = 'explore'

    def confirm_order(candidate):
        joint = candidate['floor_probabilities']['joint']
        return (
            -(joint if joint is not None else 1.0),
            candidate['predicted']['loss']['mean'],
            -candidate['predicted']['strength']['mean'],
        )

    confirms.sort(key=confirm_order)
    # Ranking pools are the plausible boundary region, not only its Pareto
    # front.  The front can be a single point when the two information measures
    # are strongly correlated, which would under-fill a multi-trial week.  Each
    # list is ordered by one objective at a time and the two lists are
    # interleaved, so no weighted composite is introduced.
    explore_strength = sorted(
        plausibly_feasible, key=lambda c: (-c['information_gain']['strength'], c['cost'])
    )
    explore_loss = sorted(
        plausibly_feasible, key=lambda c: (-c['information_gain']['loss'], c['cost'])
    )

    desired = request['n_trials']
    intent = request['intent']
    status = 'recommended'
    order = []

    if intent == 'confirm':
        order = list(confirms)
        if not order:
            return finish('no_recommendation', [], [], [
                '按“确认已有优势”的意图，没有候选能在声明置信度下满足性能底线'
                '且不被已有观测支配。可降低底线、补充证据，或改用探索意图。'
            ])
    elif intent == 'explore':
        order = _interleave(explore_strength, explore_loss)
        if not order:
            return finish('no_recommendation', [], [], [
                '按“探索未知区域”的意图，没有候选同时具备观测覆盖、安全间隔和可用的信息增益。'
            ])
    elif not declared and desired == 1:
        return finish('preference_required', [], _alternatives(performance_front, components, request['confidence'], 6), [
            '未声明强度底线或损耗上限，而强度与损耗在现有数据中互相冲突；'
            '在不引入任意权重的前提下无法选定唯一配方。请给出至少一项性能底线，'
            '或把 intent 设为 confirm/explore，或把 n_trials 设为 2 以分别覆盖两个目标。'
        ])
    elif not declared:
        # With two slots and no declared floors, spend one slot on each end of
        # the performance front; this is an explicit coverage rule, not a
        # weighted score.
        by_strength = sorted(confirms, key=lambda c: (-c['predicted']['strength']['mean'], c['predicted']['loss']['mean']))
        by_loss = sorted(confirms, key=lambda c: (c['predicted']['loss']['mean'], -c['predicted']['strength']['mean']))
        for candidate in (by_strength[0] if by_strength else None, by_loss[0] if by_loss else None):
            if candidate is not None and candidate not in order:
                order.append(candidate)
        order.extend(_interleave(explore_strength, explore_loss))
    else:
        order = list(confirms)
        if not order:
            status = 'exploratory_only'
            order = _interleave(explore_strength, explore_loss)
            if not order:
                return finish('no_recommendation', [], [], [
                    '没有候选能在声明置信度下满足性能底线，也没有同时满足安全与信息价值要求的探索候选；'
                    '建议先补充现有区域附近的试验，而不是给出未经支持的配方。'
                ])

    selected = []
    total_cost = 0.0
    for candidate in order:
        if len(selected) >= desired:
            break
        if candidate in selected:
            continue
        if any(_pairwise_distance(candidate['fractions'], other['fractions']) < request['min_separation']
               for other in selected):
            continue
        if total_cost + candidate['cost'] > request['cost_budget'] + 1e-9:
            continue
        selected.append(candidate)
        total_cost += candidate['cost']

    if not selected:
        return finish('no_recommendation', [], [], [
            f'本周成本上限 {request["cost_budget"]:g} 无法覆盖任何满足分离要求的候选；'
            '建议提高预算或减少单次试验范围。'
        ])

    if len(selected) < desired:
        reasons.append(
            f'本周只安排 {len(selected)} 次：其余候选受成本上限或信息重复（最小间距 '
            f'{request["min_separation"]:g}）限制。'
        )

    suggestions = []
    for candidate in selected:
        payload = _payload(candidate, components, request['confidence'])
        payload['rationale'] = _rationale(candidate, declared, leaders)
        suggestions.append(payload)

    alternatives = _alternatives(
        [c for c in performance_front if c not in selected],
        components,
        request['confidence'],
        6,
    )

    if status == 'exploratory_only':
        reasons.append(
            '在可制造、观测覆盖和安全间隔内，没有组合能以声明置信度同时满足性能底线；'
            '以下建议只用于减少边界附近的不确定性，不表示已经达到底线或接近最优。'
        )
    if not declared:
        reasons.append('未同时声明强度底线与损耗上限，建议中的“确认”只相对于已有观测前沿，不代表满足业务底线。')
    reasons.append(
        f'安全判断使用失败记录邻近度与局部失败占比（失败间隙 {failure_clearance:g}），'
        '这是经验性保守规则，不是通过安全认证所需的证据。'
    )
    if basis['distinct_batches'] < 6:
        reasons.append('独立炉次少于 6 个，炉次间波动估计不稳，预测区间应视为偏乐观。')
    if any(s['role'] == 'confirm' for s in suggestions) and any(s['role'] == 'explore' for s in suggestions):
        reasons.append('本次同时包含确认与探索建议：前者检验已有优势是否可复现，后者降低未知区域的判断不确定性。')

    state['budget'] = {
        'requested_trials': desired,
        'planned_trials': len(selected),
        'total_cost': float(total_cost),
        'cost_budget': request['cost_budget'],
    }
    return finish(status, suggestions, alternatives, [])


def _interleave(first, second):
    merged = []
    for index in range(max(len(first), len(second))):
        for rows in (first, second):
            if index < len(rows) and rows[index] not in merged:
                merged.append(rows[index])
    return merged


def _rationale(candidate, declared, leaders):
    strength = candidate['predicted']['strength']['mean']
    loss = candidate['predicted']['loss']['mean']
    where = '已完成试验覆盖范围内' if candidate['support']['inside_observed_ranges'] else '已观测范围之外、仍在探索裕度内'
    if candidate['role'] == 'confirm':
        strength_leader = leaders['strength']
        loss_leader = leaders['loss']
        text = (
            f'确认已有优势：组合位于{where}，预计强度 {strength:.4g}、损耗 {loss:.4g}。'
            f'现有最强完成记录 {strength_leader["id"]} 为强度 {strength_leader["strength"]:.4g}、'
            f'损耗 {strength_leader["loss"]:.4g}；现有最低损耗记录 {loss_leader["id"]} 为'
            f'损耗 {loss_leader["loss"]:.4g}、强度 {loss_leader["strength"]:.4g}。'
        )
        joint = candidate['floor_probabilities']['joint']
        if joint is not None:
            text += f'以声明置信度满足全部底线的概率为 {joint:.2f}；该概率来自当前证据，不是保证。'
        else:
            text += '该组合不被任何已完成试验的观测结果支配，但仍需用新试验确认。'
        return text
    distance = candidate['support']['distance_to_nearest_completed']
    text = (
        f'探索未知区域：组合位于{where}，与最近已完成配方的距离为 {distance:.3g}，'
        f'预计强度 {strength:.4g}、损耗 {loss:.4g}；'
        f'对强度与损耗的信息增益分别为 {candidate["information_gain"]["strength"]:.3f} 与 '
        f'{candidate["information_gain"]["loss"]:.3f} 纳特。'
        '该建议用于减少边界附近的不确定性，尚不足以确认满足性能底线。'
    )
    if declared and candidate['floor_probabilities']['joint'] is not None:
        text += f'当前证据下满足全部底线的概率仅 {candidate["floor_probabilities"]["joint"]:.2f}。'
    return text


def _alternatives(candidates, components, confidence, limit):
    if not candidates:
        return []
    picks = []

    def add(candidate):
        if candidate is not None and candidate not in picks:
            picks.append(candidate)

    add(max(candidates, key=lambda c: c['predicted']['strength']['mean']))
    add(min(candidates, key=lambda c: c['predicted']['loss']['mean']))
    add(max(candidates, key=lambda c: c['information_gain']['strength']))
    add(max(candidates, key=lambda c: c['information_gain']['loss']))
    if candidates[0]['floor_probabilities']['joint'] is not None:
        add(max(candidates, key=lambda c: c['floor_probabilities']['joint']))
    for candidate in candidates:
        if len(picks) >= limit:
            break
        add(candidate)
    return [_payload(candidate, components, confidence) for candidate in picks[:limit]]
