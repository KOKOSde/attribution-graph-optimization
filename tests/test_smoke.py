import os
import unittest

import torch

from attribution_graph_optimization import extract_features, get_native_extension_status


class SmokeTest(unittest.TestCase):
    def test_cpu_fallback_import_and_execution(self):
        os.environ['ATTR_GRAPH_DISABLE_EXTENSION'] = '1'
        feat_acts = torch.tensor(
            [
                [
                    [0.10, 0.40, 0.05, 0.90],
                    [0.20, 0.80, 0.03, 0.70],
                ]
            ],
            dtype=torch.float32,
        )

        nodes = extract_features(
            feat_acts=feat_acts,
            layer_idx=7,
            top_k=2,
            threshold=0.15,
            implementation='auto',
        )

        status = get_native_extension_status()
        self.assertFalse(status['enabled'])
        self.assertEqual(len(nodes), 4)
        self.assertEqual(nodes[0]['layer'], '7')
        self.assertEqual(nodes[0]['ctx_idx'], 0)
        self.assertIn('activation', nodes[0])

        del os.environ['ATTR_GRAPH_DISABLE_EXTENSION']


if __name__ == '__main__':
    unittest.main()
