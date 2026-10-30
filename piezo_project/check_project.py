from itertools import combinations, product
from pathlib import Path
import json
import unittest
import numpy as np
from scipy import stats
import piezo_data as P
import statlib as S

ROOT = Path(__file__).resolve().parent

class ProjectChecks(unittest.TestCase):
    def test_formula_groups_and_repeated_elements(self):
        self.assertEqual(P.elems('Ni(HO)2'), {'Ni': 1, 'H': 2, 'O': 2})
        self.assertEqual(P.elems('Ca3(PO4)2'), {'Ca': 3, 'P': 2, 'O': 8})
        self.assertEqual(P.elems('CH3COOH'), {'C': 2, 'H': 4, 'O': 2})
        self.assertEqual(P.elems('K4(ON(SO3)2)2'), {'K': 4, 'O': 14, 'N': 2, 'S': 4})

    def test_invalid_formula_fails(self):
        for formula in ['Ca(OH', 'Ca)OH', 'Na2#Cl', '2NaCl']:
            with self.assertRaises(ValueError):
                P.elems(formula)

    def test_holm_known_example(self):
        np.testing.assert_allclose(S.holm_adjust([0.04, 0.001, 0.03]), [0.06, 0.003, 0.06])
        np.testing.assert_allclose(S.holm_adjust([0.8, 0.9]), [1, 1])

    def test_finite_population_se_exact_enumeration(self):
        pop = np.array([1., 2., 5., 9.])
        m = 2
        for replace in [False, True]:
            samples = list(product(pop, repeat=m)) if replace else list(combinations(pop, m))
            means = np.mean(samples, axis=1)
            exact = means.std(ddof=0)
            result = S.sampling_experiment(pop, m, reps=10, replace=replace)
            self.assertAlmostEqual(result['SE(x̄) теор.'], exact, places=12)

    def test_conditional_lomax_mle(self):
        x = np.array([0.05, 0.2, 0.4, 1., 2.])
        scale = 0.5
        shape = len(x) / np.log1p(x / scale).sum()
        ll = lambda a: stats.lomax.logpdf(x, a, loc=0, scale=scale).sum()
        self.assertGreater(ll(shape), ll(shape * 0.99))
        self.assertGreater(ll(shape), ll(shape * 1.01))

    def test_notebook_scope_and_execution(self):
        nb = json.loads((ROOT / 'lab01_sprint1_piezo.ipynb').read_text())
        text = '\n'.join(''.join(c['source']) for c in nb['cells'])
        for removed in ['RV03', 'x3', '#rv3', 'NIMS', '✎', 'напиши своими словами']:
            self.assertNotIn(removed, text)
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                self.assertIsInstance(c['execution_count'], int)
                self.assertFalse(any(o['output_type'] == 'error' for o in c['outputs']))

    def test_summary_matches_raw_data(self):
        import pandas as pd
        raw = json.loads((ROOT / 'data/piezo.json').read_text())
        x1 = np.array([row['eij_max'] for row in raw])
        x2 = pd.read_csv(ROOT / 'data/matbench_dielectric_n_folds.csv').n.to_numpy()
        summary = pd.read_csv(ROOT / 'summary_metrics.csv', index_col=0)
        for name, values in [('e_ij_max', x1), ('n', x2)]:
            self.assertEqual(int(summary.loc[name, 'N']), len(values))
            expected = [values.mean(), values.std(ddof=1), np.median(values),
                        stats.skew(values), stats.kurtosis(values)]
            np.testing.assert_allclose(summary.loc[name, ['mean', 'SD', 'median', 'skew', 'ex.kurt']].astype(float), expected, rtol=1e-12)

    def test_hypotheses_match_raw_data(self):
        import pandas as pd
        from scipy.special import logsumexp, gammaln
        d = pd.read_csv(ROOT / 'data/matbench_dielectric_n_folds.csv')
        h = pd.read_csv(ROOT / 'hypotheses.csv').set_index('гипотеза')
        self.assertEqual(len(h), 11)
        positive = int((d.n > d.n_pred_modnet).sum())
        total = int((d.n != d.n_pred_modnet).sum())
        # Exact symmetric binomial tails, independently summed in log space.
        k = np.arange(min(positive, total - positive) + 1)
        logpmf = gammaln(total + 1) - gammaln(k + 1) - gammaln(total - k + 1) - total * np.log(2)
        exact_p = min(1., 2 * np.exp(logsumexp(logpmf)))
        self.assertAlmostEqual(h.loc['H2.5', 'p'], exact_p, places=11)
        self.assertAlmostEqual(h.loc['H2.5', 'статистика'], positive / total, places=12)
        medians = d.groupby('fold').n.median()
        self.assertIn(f'{medians.max() - medians.min():.4f}', h.loc['H2.3', 'эффект'])
        np.testing.assert_allclose(h['p (Холм)'], S.holm_adjust(h.p), rtol=1e-12)
        self.assertFalse(h['эффект'].isna().any())

    def test_threshold_text_matches_experiment(self):
        nb = json.loads((ROOT / 'lab01_sprint1_piezo.ipynb').read_text())
        text = '\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertNotIn('0.1σ', text)
        self.assertNotIn('0.1 * pop.std', text)
        text_outputs = '\n'.join(''.join(o.get('text', [])) for c in nb['cells'] for o in c.get('outputs', []))
        self.assertNotIn('0.1σ', text_outputs)

if __name__ == '__main__':
    unittest.main(verbosity=2)
