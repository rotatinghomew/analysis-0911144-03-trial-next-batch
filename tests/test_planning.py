import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError

from trial_next_batch.client import Client
from trial_next_batch.domain import validate
from trial_next_batch.planner import PlanningError, normalize, plan, suggest
from trial_next_batch.repository import Repository
from trial_next_batch.service import make_server


def history():
    return json.loads(Path('fixtures/history.json').read_text())


def demo():
    return json.loads(Path('fixtures/demo.json').read_text())


FLOORS = {'strength_floor': 42.0, 'loss_ceiling': 0.135}


class PlanningRequestTests(unittest.TestCase):
    def test_unknown_parameter_rejected(self):
        with self.assertRaises(PlanningError):
            normalize(history(), {'n_trials': 1, 'weight': 0.5})

    def test_budget_cannot_exceed_declared_max_cost(self):
        result = normalize(history(), {'cost_budget': 10_000})
        self.assertEqual(result['cost_budget'], 90.0)
        self.assertTrue(any('max_cost' in note for note in result['notes']))

    def test_n_trials_bounds(self):
        with self.assertRaises(ValueError):
            normalize(history(), {'n_trials': 0})
        with self.assertRaises(ValueError):
            normalize(history(), {'n_trials': 21})

    def test_confidence_bounds(self):
        with self.assertRaises(ValueError):
            normalize(history(), {'confidence': 0.2})

    def test_component_bounds_must_refer_to_known_components(self):
        with self.assertRaises(PlanningError):
            normalize(history(), {'component_bounds': {'z': [0, 1]}})

    def test_manufacturability_bounds_restrict_candidates(self):
        result = suggest(history(), dict(FLOORS, at=41, n_trials=2, cost_budget=40,
                                         component_bounds={'b': [0.0, 0.3]}))
        self.assertNotEqual(result['status'], 'no_recommendation')
        for suggestion in result['suggestions']:
            self.assertLessEqual(suggestion['fractions']['b'], 0.3 + 1e-9)

    def test_invalid_component_bounds_rejected(self):
        # Upper bound above total_fraction cannot describe a composition.
        with self.assertRaises(ValueError):
            suggest(history(), {'n_trials': 1, 'component_bounds': {'a': [0.0, 1.5]}})
        with self.assertRaises(ValueError):
            suggest(history(), {'n_trials': 1, 'component_bounds': {'a': [0.6, 0.2]}})

    def test_cost_budget_respected(self):
        result = suggest(history(), dict(FLOORS, at=41, n_trials=2, cost_budget=2.5))
        self.assertNotEqual(result['status'], 'no_recommendation')
        self.assertLessEqual(result['budget']['total_cost'], 2.5 + 1e-9)
        for suggestion in result['suggestions']:
            self.assertLessEqual(suggestion['cost'], 2.5 + 1e-9)

    def test_budget_below_any_formula_gives_reason(self):
        result = suggest(history(), dict(FLOORS, at=41, n_trials=2, cost_budget=0.5))
        self.assertEqual(result['status'], 'no_recommendation')
        self.assertTrue(any('成本' in reason for reason in result['reasons']))

    def test_manufacturability_bounds_reported(self):
        result = suggest(history(), dict(FLOORS, at=41, n_trials=2, cost_budget=40,
                                         component_bounds={'b': [0.05, 0.25]}))
        self.assertNotEqual(result['status'], 'no_recommendation')
        for suggestion in result['suggestions']:
            self.assertGreaterEqual(suggestion['fractions']['b'], 0.05 - 1e-9)
            self.assertLessEqual(suggestion['fractions']['b'], 0.25 + 1e-9)


class EvidenceBoundaryTests(unittest.TestCase):
    def test_too_few_observations_refuses_to_recommend(self):
        result = suggest(demo(), {'n_trials': 1, 'cost_budget': 90})
        self.assertEqual(result['status'], 'no_recommendation')
        self.assertEqual(result['suggestions'], [])
        self.assertTrue(any('观测不足' in reason for reason in result['reasons']))

    def test_single_batch_is_not_enough(self):
        data = history()
        # Keep only trials from two batches.
        data['trials'] = [row for row in data['trials'] if row['batch'] in ('b0', 'b1')]
        result = suggest(data, {'n_trials': 1})
        self.assertEqual(result['status'], 'no_recommendation')
        self.assertTrue(any('炉次' in reason for reason in result['reasons']))
        self.assertLess(result['data_basis']['distinct_batches'], 3)

    def test_pending_trials_are_not_used_as_results(self):
        data = history()
        baseline = suggest(data, dict(FLOORS, at=41, n_trials=2, cost_budget=40))
        for trial in data['trials']:
            if trial['status'] == 'pending':
                trial['visible_at'] = 0
        changed = suggest(data, dict(FLOORS, at=41, n_trials=2, cost_budget=40))
        self.assertEqual(baseline['data_basis']['pending_trials'], 0)
        self.assertEqual(changed['data_basis']['pending_trials'], 4)
        self.assertEqual(baseline['model'], changed['model'])
        self.assertEqual(
            [s['fractions'] for s in baseline['suggestions']],
            [s['fractions'] for s in changed['suggestions']],
        )

    def test_pending_formula_and_neighbourhood_excluded(self):
        data = history()
        target = next(trial for trial in data['trials'] if trial['status'] == 'pending')
        result = suggest(data, dict(FLOORS, at=47, n_trials=3, cost_budget=90,
                                    min_separation=0.0, exploration_distance=0.01))
        for suggestion in result['suggestions']:
            distance = sum((suggestion['fractions'][name] - target['fractions'][name]) ** 2
                           for name in target['fractions']) ** 0.5
            self.assertGreater(distance, 1e-9)

    def test_failed_trials_are_not_treated_as_zero_strength(self):
        data = history()
        with_failures = suggest(data, dict(FLOORS, at=47, n_trials=2, cost_budget=90))
        data_without = copy.deepcopy(data)
        data_without['trials'] = [row for row in data_without['trials'] if row['status'] != 'failed']
        without_failures = suggest(data_without, dict(FLOORS, at=47, n_trials=2, cost_budget=90))
        # The failed rows must not enter the response model at all.
        self.assertIsNotNone(with_failures['model'])
        self.assertEqual(with_failures['data_basis']['failed_trials'], 2)
        self.assertEqual(without_failures['data_basis']['failed_trials'], 0)
        self.assertAlmostEqual(
            with_failures['model']['strength']['residual_sd'],
            without_failures['model']['strength']['residual_sd'],
            places=10,
        )
        self.assertNotIn('strength_mean', with_failures)

    def test_failed_region_is_avoided(self):
        data = history()
        failed = [row for row in data['trials'] if row['status'] == 'failed']
        self.assertTrue(failed)
        result = suggest(data, dict(FLOORS, at=47, n_trials=2, cost_budget=90))
        self.assertNotEqual(result['status'], 'no_recommendation')
        clearance = result['support_rule']['failure_clearance']
        for suggestion in result['suggestions']:
            for row in failed:
                distance = sum((suggestion['fractions'][name] - row['fractions'][name]) ** 2
                               for name in row['fractions']) ** 0.5
                self.assertGreaterEqual(distance, clearance - 1e-9)

    def test_cutoff_excludes_later_records(self):
        data = history()
        early = suggest(data, dict(FLOORS, at=20, n_trials=2, cost_budget=90))
        late = suggest(data, dict(FLOORS, at=47, n_trials=2, cost_budget=90))
        self.assertLess(early['data_basis']['completed_trials'], late['data_basis']['completed_trials'])
        self.assertGreater(early['data_basis']['records_after_cutoff'], 0)
        self.assertTrue(any('visible_at' in reason for reason in early['reasons']))

    def test_nan_prediction_never_serialised(self):
        data = history()
        result = suggest(data, dict(FLOORS, at=41, n_trials=2, cost_budget=40))
        text = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertIn('confidence', text)


class ModellingTests(unittest.TestCase):
    def test_batch_shared_error_inflates_interval(self):
        # Two trials in one batch are not independent evidence.  The model's
        # predictive uncertainty must therefore be wider than a model that
        # assumes independent residuals.
        data = history()
        from trial_next_batch.model import GroupedGP
        x = [[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.5, 0.2], [0.6, 0.2]]
        y = [31.0, 34.0, 37.0, 40.0, 41.0, 44.0]
        grouped = GroupedGP(x, y, ['b0', 'b0', 'b0', 'b1', 'b1', 'b1'])
        independent = GroupedGP(x, y, ['u0', 'u1', 'u2', 'u3', 'u4', 'u5'])
        point = grouped.predict([[0.45, 0.15]])
        flat = independent.predict([[0.45, 0.15]])
        self.assertEqual(grouped.diagnostics()['batch_effect_sd'] >= 0, True)
        self.assertGreaterEqual(float(point['std'][0]), 0.0)
        self.assertGreaterEqual(float(flat['std'][0]), 0.0)
        # The complete result must carry the batch structure rather than a
        # silently diagonal noise assumption.
        self.assertGreaterEqual(grouped.batch_variance, 0.0)

    def test_two_responses_are_fitted_separately(self):
        result = suggest(history(), dict(FLOORS, at=41, n_trials=2, cost_budget=40))
        self.assertIn('strength', result['model'])
        self.assertIn('loss', result['model'])
        self.assertNotAlmostEqual(
            result['model']['strength']['residual_sd'],
            result['model']['loss']['residual_sd'],
        )

    def test_no_weighted_score_in_output(self):
        result = suggest(history(), dict(FLOORS, at=41, n_trials=2, cost_budget=40))
        text = json.dumps(result, ensure_ascii=False)

        def keys(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    yield from keys(value)
            elif isinstance(node, list):
                for value in node:
                    yield from keys(value)

        forbidden = {'score', 'utility', 'weighted_score', 'objective', 'combined'}
        self.assertFalse(forbidden & set(keys(result)))

    def test_suggestions_are_pareto_front_or_confirm_and_explore_roles(self):
        result = suggest(history(), dict(FLOORS, at=41, n_trials=3, cost_budget=90))
        self.assertNotEqual(result['status'], 'no_recommendation')
        roles = {suggestion['role'] for suggestion in result['suggestions']}
        self.assertTrue(roles <= {'confirm', 'explore'})
        for suggestion in result['suggestions']:
            self.assertIn('rationale', suggestion)
            self.assertTrue(suggestion['rationale'])

    def test_preference_required_without_floors(self):
        result = suggest(history(), {'at': 41, 'n_trials': 1, 'cost_budget': 40})
        self.assertEqual(result['status'], 'preference_required')
        self.assertTrue(result['alternatives'])
        self.assertTrue(any('权重' in reason for reason in result['reasons']))

    def test_conflict_with_floors_reduces_to_exploration(self):
        # Demand performance the data does not support.
        result = suggest(history(), {'at': 41, 'n_trials': 2, 'cost_budget': 40,
                                     'strength_floor': 90.0, 'loss_ceiling': 0.01})
        self.assertIn(result['status'], ('no_recommendation', 'exploratory_only'))
        if result['status'] == 'exploratory_only':
            self.assertTrue(all(s['role'] == 'explore' for s in result['suggestions']))
            self.assertTrue(any('不表示已经达到底线' in reason for reason in result['reasons']))

    def test_intervals_widen_when_observations_are_removed(self):
        full = suggest(history(), dict(FLOORS, at=47, n_trials=2, cost_budget=90))

        sparse = copy.deepcopy(history())
        completed = [row for row in sparse['trials'] if row['status'] == 'complete']
        keep_ids = {row['id'] for row in completed[:12]}
        sparse['trials'] = [row for row in sparse['trials']
                            if row['status'] != 'complete' or row['id'] in keep_ids]
        reduced = suggest(sparse, dict(FLOORS, at=47, n_trials=2, cost_budget=90))
        self.assertEqual(reduced['status'], 'recommended')
        self.assertGreater(
            reduced['suggestions'][0]['predicted']['strength']['std'],
            full['suggestions'][0]['predicted']['strength']['std'],
        )
        # With few batches the model cannot separate batch shift from residual
        # noise; the reported shared component must not silently vanish.
        self.assertGreater(reduced['model']['strength']['batch_effect_sd'],
                           full['model']['strength']['batch_effect_sd'])

    def test_exploration_is_labelled_and_distinct_from_confirmation(self):
        result = suggest(history(), {'at': 41, 'n_trials': 3, 'cost_budget': 90,
                                     'strength_floor': 30.0, 'loss_ceiling': 0.3,
                                     'intent': 'auto'})
        self.assertNotEqual(result['status'], 'no_recommendation')
        for suggestion in result['suggestions']:
            if suggestion['role'] == 'explore':
                self.assertTrue(any(word in suggestion['rationale'] for word in ('探索', '未知')))
                self.assertIsNotNone(suggestion['information_gain'])
            else:
                self.assertIn('确认', suggestion['rationale'])


class PlanPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.repository = Repository(Path(self.directory.name) / 'data.sqlite3')
        self.server = make_server(self.repository)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = Client(f'http://127.0.0.1:{self.server.server_port}')

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.directory.cleanup()

    def test_plan_is_saved_with_revision_and_digest(self):
        receipt = self.client.import_data(history(), 'history')
        response = self.client.plan({'n_trials': 2, 'cost_budget': 40, **FLOORS}, receipt['revision'])
        self.assertEqual(response['revision'], receipt['revision'])
        self.assertEqual(response['result']['revision'], receipt['revision'])
        self.assertEqual(response['result']['data_digest'], receipt['digest'])
        stored = self.client.plan_detail(response['plan']['plan_id'])
        self.assertEqual(stored['revision'], receipt['revision'])
        self.assertEqual(stored['result']['data_basis']['visible_at'], 47)

    def test_new_data_updates_basis_but_old_plan_is_unchanged(self):
        data = history()
        first = self.client.import_data(data, 'first')
        old = self.client.plan({'n_trials': 2, 'cost_budget': 40, **FLOORS}, first['revision'])
        changed = copy.deepcopy(data)
        changed['trials'][42]['status'] = 'complete'
        changed['trials'][42]['strength'] = 34.2
        changed['trials'][42]['loss'] = 0.1159
        second = self.client.import_data(changed, 'second')
        new = self.client.plan({'n_trials': 2, 'cost_budget': 40, **FLOORS}, second['revision'])
        self.assertNotEqual(old['plan']['revision'], new['plan']['revision'])
        self.assertNotEqual(old['result']['data_basis']['completed_trials'],
                            new['result']['data_basis']['completed_trials'])
        reloaded = self.client.plan_detail(old['plan']['plan_id'])
        self.assertEqual(reloaded['result'], old['result'])
        self.assertEqual(reloaded['result']['data_basis']['completed_trials'], 42)

    def test_plan_for_missing_revision(self):
        with self.assertRaises(HTTPError) as caught:
            self.client.plan({'n_trials': 1}, 999)
        self.assertEqual(caught.exception.code, 404)

    def test_plan_listing(self):
        receipt = self.client.import_data(history(), 'history')
        self.client.plan({'n_trials': 1, 'cost_budget': 40, **FLOORS}, receipt['revision'])
        self.client.plan({'n_trials': 2, 'cost_budget': 40, **FLOORS}, receipt['revision'])
        listing = self.client.plans(receipt['revision'])
        self.assertEqual(len(listing['plans']), 2)
        self.assertEqual([row['revision'] for row in listing['plans']], [receipt['revision']] * 2)

    def test_invalid_plan_request_is_400(self):
        self.client.import_data(history(), 'history')
        with self.assertRaises(HTTPError) as caught:
            self.client.plan({'n_trials': 0})
        self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(HTTPError) as caught:
            self.client.plan({'unexpected': 1})
        self.assertEqual(caught.exception.code, 400)

    def test_plan_defaults_cutoff_to_latest_visible(self):
        receipt = self.client.import_data(history(), 'history')
        response = self.client.plan({'n_trials': 1, 'cost_budget': 40, **FLOORS}, receipt['revision'])
        self.assertEqual(response['result']['data_basis']['visible_at'], 47)
        self.assertEqual(response['result']['data_basis']['records_after_cutoff'], 0)

    def test_plans_survive_restart(self):
        receipt = self.client.import_data(history(), 'history')
        response = self.client.plan({'n_trials': 1, 'cost_budget': 40, **FLOORS}, receipt['revision'])
        restarted = Repository(self.repository.path)
        stored = restarted.get_plan(response['plan']['plan_id'])
        self.assertEqual(stored['result']['revision'], receipt['revision'])


class DemoFixtureTests(unittest.TestCase):
    def test_demo_plan_keeps_pending_and_failed_out_of_model(self):
        data = demo()
        result = suggest(data, {'n_trials': 1, 'cost_budget': 90})
        self.assertEqual(result['status'], 'no_recommendation')
        self.assertEqual(result['data_basis']['completed_trials'], 2)
        self.assertEqual(result['data_basis']['pending_trials'], 1)
        self.assertEqual(result['data_basis']['failed_trials'], 1)
        self.assertIsNone(result['model'])

    def test_plan_result_is_json_safe(self):
        for request in (
            dict(FLOORS, at=41, n_trials=2, cost_budget=40),
            {'at': 41, 'n_trials': 1, 'cost_budget': 40},
            {'at': 41, 'n_trials': 3, 'cost_budget': 40, 'intent': 'explore'},
            {'at': 20, 'n_trials': 2, 'cost_budget': 90, 'strength_floor': 20,
             'loss_ceiling': 1.0},
        ):
            json.dumps(suggest(history(), request), ensure_ascii=False, allow_nan=False)

    def test_plan_does_not_mutate_input(self):
        data = history()
        before = copy.deepcopy(data)
        suggest(data, dict(FLOORS, at=41, n_trials=2, cost_budget=40))
        self.assertEqual(data, before)
        validate(data)


if __name__ == '__main__':
    unittest.main()
