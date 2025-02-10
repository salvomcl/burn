use core::f64;
use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor};

/// The Area Under the Receiver Operating Characteristic Curve (AUROC, also referred to as [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)) for binary classification.
#[derive(Default)]
pub struct AveragePrecisionMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
}

/// The [AUROC metric](AurocMetric) input type.
#[derive(new)]
pub struct AveragePrecisionInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> AveragePrecisionMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    fn binary_average_precision(
        &self,
        probabilities: &Tensor<B, 1>,
        targets: &Tensor<B, 1, Int>,
    ) -> f64 {
        let n = targets.dims()[0];

        let n_pos = targets.clone().sum().into_scalar().elem::<u64>() as usize;

        // Early return if we don't have both positive and negative samples
        if n_pos == 0 || n_pos == n {
            if n_pos == 0 {
                log::warn!("Metric cannot be computed because all target values are negative.")
            } else {
                log::warn!("Metric cannot be computed because all target values are positive.")
            }
            return 0.0;
        }

        let pos_mask = targets.clone().equal_elem(1).int().reshape([n, 1]);
        let neg_mask = targets.clone().equal_elem(0).int().reshape([1, n]);

        let valid_pairs = pos_mask * neg_mask;

        let prob_i = probabilities.clone().reshape([n, 1]).repeat_dim(1, n);
        let prob_j = probabilities.clone().reshape([1, n]).repeat_dim(0, n);

        let correct_order = prob_i.clone().greater(prob_j.clone()).int();

        let ties = prob_i.equal(prob_j).int();

        // Calculate AUC components
        let num_pairs = valid_pairs.clone().sum().into_scalar().elem::<f64>();
        let correct_pairs = (correct_order * valid_pairs.clone())
            .sum()
            .into_scalar()
            .elem::<f64>();
        let tied_pairs = (ties * valid_pairs).sum().into_scalar().elem::<f64>();

        (correct_pairs + 0.5 * tied_pairs) / num_pairs
    }
}

impl<B: Backend> Metric for AveragePrecisionMetric<B> {
    const NAME: &'static str = "Area Under the Receiver Operating Characteristic Curve";
    type Input = AveragePrecisionInput<B>;

    fn update(
        &mut self,
        input: &AveragePrecisionInput<B>,
        _metadata: &MetricMetadata,
    ) -> MetricEntry {
        let [batch_size, num_classes] = input.outputs.dims();

        assert_eq!(
            num_classes, 2,
            "Currently only binary classification is supported"
        );

        let probabilities = {
            let exponents = input.outputs.clone().exp();
            let sum = exponents.clone().sum_dim(1);
            (exponents / sum)
                .select(1, Tensor::arange(1..2, &input.outputs.device()))
                .squeeze(1)
        };

        let area_under_curve = self.binary_average_precision(&probabilities, &input.targets);

        self.state.update(
            100.0 * area_under_curve,
            batch_size,
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for AveragePrecisionMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_auroc() {
        let device = Default::default();

        let preds: Tensor<TestBackend, 1> = Tensor::from_data(
            [
                0.6415, 0.9519, 0.9603, 0.0432, 0.9516, 0.6263, 0.6666, 0.1332, 0.0507, 0.9452,
            ],
            &device,
        );
        let target: Tensor<TestBackend, 1, Int> =
            Tensor::from_data([1, 0, 0, 1, 1, 0, 1, 0, 1, 1], &device);

        let [n_samples] = preds.dims();

        let desc_score_indices = preds.clone().argsort_descending(0);

        let target = target.clone().select(0, desc_score_indices.clone());

        let threshold_idxs = Tensor::arange(0..(n_samples as i64), &device).float();
        let tps = target.clone().cumsum(0).float();
        let fps = threshold_idxs - tps.clone() + 1;

        let precision = tps.clone().div(tps.clone() + fps);
        let positives = target.sum().float();
        let recall = tps.div(positives);
        let precision = precision.flip([0]);
        let recall = recall.flip([0]);

        let recall_cur = recall.clone().slice([0..n_samples - 1]);
        let recall_next = recall.slice([1..n_samples]);
        let precision_cur = precision.slice([0..n_samples - 1]);
        let average_precision = -((recall_next - recall_cur) * precision_cur)
            .sum()
            .into_scalar(); //TODO f32 or f64?
        dbg!(average_precision);
    }
}
