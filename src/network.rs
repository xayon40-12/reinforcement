use std::ops::RangeInclusive;

use activation::{Activation, BoundedActivation};

pub mod activation;
pub mod layer;
pub mod layers;
pub mod reinforcement;

pub type Float = f64;

pub trait JoinNetwork<const NI: usize> {
    type OutA: Activation;
}
pub trait ForwardNetwork<const NI: usize, const NO: usize>: JoinNetwork<NI> {
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO];
}
pub trait Network<const NI: usize, const NO: usize>: ForwardNetwork<NI, NO> {
    fn randomize(&mut self);
    /// The parameter $r$ is the ratio $0 < r <= 1$ for the relaxation of the update of the gradient. A value of $r = 1$ correspond to keep only the new value of the gradient, whereas $r = 0.5$ will average the new and last value.
    fn update_gradient(&mut self, relaxation: Float, delta: [Float; NO]) -> [Float; NI];
    fn reset_gradient(&mut self);
    fn apply_gradient(&mut self, alpha: Float);
    fn rescale_gradient(&mut self, a: Float);
    fn norm2_gradient(&self) -> Float;
    fn normalize_gradient(&mut self) {
        self.rescale_gradient(self.norm2_gradient().sqrt().recip());
    }
    fn train(
        &mut self,
        iterations: usize,
        alpha: Float,
        training_data: &[([Float; NI], [Float; NO])],
    ) {
        let align = iterations.ilog10() as usize;
        for i in 0..iterations {
            let mut error = 0.0;
            for &(inputs, targets) in training_data.iter() {
                let outputs = self.forward(inputs);
                if i % (iterations / 10) == 0 {
                    print!("{outputs:?} ");
                }
                let normalization = (2.0 * NO as Float).recip();
                let delta = std::array::from_fn(|i| normalization * (targets[i] - outputs[i]));
                error += delta.iter().map(|d| d * d).sum::<Float>();
                self.update_gradient(1.0, delta);
                self.normalize_gradient();
                self.apply_gradient(alpha);
            }
            if i % (iterations / 10) == 0 {
                println!("\n{i:<align$}: error = {:.1e}", error.sqrt());
            }
        }
    }
}
pub trait BoundedNetwork<const NI: usize, const NO: usize>:
    Network<NI, NO, OutA: BoundedActivation>
{
    fn output_ranges(&self) -> [RangeInclusive<Float>; NO];
}
pub trait StochasticForwardNetwork<const NI: usize, const NO: usize>:
    JoinNetwork<NI, OutA: BoundedActivation>
{
    fn pert_forward(&mut self, input: [Float; NI], shape: Float) -> [Float; NO];
}
