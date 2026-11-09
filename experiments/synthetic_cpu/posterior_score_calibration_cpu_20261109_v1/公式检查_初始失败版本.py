#!/usr/bin/env python3
"""已知后验诊断脚本的定向公式检查。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import run as experiment
    from run import enumerate_posterior_components, mixture_crps, mixture_quantile, normal_cdf
except ImportError as exc:
    print(f"预期失败：诊断实现尚未建立（{exc}）")
    raise SystemExit(1) from exc


class 后验评分公式检查(unittest.TestCase):
    def test_实验维度与预登记协议一致(self):
        self.assertEqual(experiment.DIMENSION, 6)
        self.assertEqual(1 << experiment.DIMENSION, 64)

    def test_正态分布函数已知值(self):
        actual = normal_cdf([0.0, 1.0, -1.0])
        expected = [0.5, 0.8413447461, 0.1586552539]
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(float(got), want, delta=2e-7)

    def test_等方差混合先验的线性高斯后验(self):
        weights, means, covariances = enumerate_posterior_components(
            observation=[0.75],
            operator=[[1.0]],
            spike_std=1.0,
            slab_std=1.0,
            inclusion_probability=0.2,
            noise_std=0.5,
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        for mean in means[:, 0]:
            self.assertAlmostEqual(float(mean), 0.6, places=12)
        for covariance in covariances[:, 0, 0]:
            self.assertAlmostEqual(float(covariance), 0.2, places=12)

    def test_单高斯混合的_crps_符合闭式结果(self):
        score = mixture_crps(
            weights=[1.0], means=[[0.0]], variances=[[1.0]], target=[0.0]
        )
        self.assertAlmostEqual(float(score), 0.2336949773, delta=2e-7)

    def test_退化点预测的_crps_等于绝对误差(self):
        score = mixture_crps(
            weights=[1.0], means=[[1.0]], variances=[[0.0]], target=[3.0]
        )
        self.assertAlmostEqual(float(score), 2.0, places=12)

    def test_单高斯混合的中位数和九十五分位数(self):
        quantiles = mixture_quantile(
            weights=[1.0], means=[[0.0]], stds=[[1.0]], probabilities=[0.5, 0.95]
        )
        self.assertAlmostEqual(float(quantiles[0, 0]), 0.0, delta=2e-7)
        self.assertAlmostEqual(float(quantiles[1, 0]), 1.6448536269, delta=2e-5)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(后验评分公式检查)
    result = unittest.TestResult()
    suite.run(result)
    if result.wasSuccessful():
        print(f"公式检查通过：{result.testsRun} 项")
    else:
        for test, _ in [*result.failures, *result.errors]:
            print(f"检查失败：{test._testMethodName}")
        raise SystemExit(1)
