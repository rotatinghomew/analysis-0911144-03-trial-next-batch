from collections import Counter
from .common import finite, require, table, unique


def validate(data):
    trials = table(data, 'trials')
    unique(trials)
    constraints = data['constraints']
    components = set(constraints['components'])
    require(components == set(constraints['costs']), '组分成本不完整')
    finite(constraints['max_cost'], 'max_cost', 0)
    for value in constraints['costs'].values():
        finite(value, 'cost', 0)
    for trial in trials:
        require(set(trial['fractions']) == components, '组分不完整')
        for value in trial['fractions'].values():
            finite(value, 'fraction', 0)
        require(abs(sum(trial['fractions'].values()) - constraints['total_fraction']) < 1e-8, '投料比例不守恒')
        require(trial['status'] in ('complete', 'pending', 'failed'), '试验状态错误')
        require(type(trial['visible_at']) is int, '可见时间必须为整数')
        require(bool(trial['batch']), '炉次不能为空')
        if trial['status'] == 'complete':
            finite(trial['strength'], 'strength')
            finite(trial['loss'], 'loss', 0)
        else:
            require(trial['strength'] is None and trial['loss'] is None, '未完成试验不能有完整结果')


def observed(data, at=None):
    return [row for row in data['trials'] if (at is None or row['visible_at'] <= at)]


def records(data):
    return data['trials']


def summarize(data):
    rows = observed(data)
    completed = [row for row in rows if row['status'] == 'complete']
    return {'states': dict(Counter(row['status'] for row in rows)), 'batches': len({row['batch'] for row in rows}), 'observed_strength_mean': None if not completed else sum(row['strength'] for row in completed) / len(completed), 'meaning': '观测描述，不是配方优选或因果结论'}
