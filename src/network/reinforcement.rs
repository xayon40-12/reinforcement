use std::ops::RangeInclusive;

use array_vector_space::ArrayVectorSpace;

use super::{Float, ForwardNetwork, JoinNetwork, Network, activation::Sigmoid};

#[derive(Default)]
pub struct Reinforcement<const NI: usize, const NO: usize, N: Network<NI, NO, OutA = Sigmoid>> {
    network: N,
    r: Float,
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO, OutA = Sigmoid>>
    Reinforcement<NI, NO, N>
{
    pub fn reinforce(
        &mut self,
        r: Float,
        mut f: impl FnMut(&mut dyn ForwardNetwork<NI, NO, OutA = Sigmoid>) -> Option<Float>,
    ) where
        Self: Sized,
    {
        self.r = r;
        loop {
            let alpha = f(self);
            if let Some(alpha) = alpha {
                self.normalize_gradient();
                self.apply_gradient(alpha);
                self.reset_gradient();
                break;
            }
        }
    }
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO, OutA = Sigmoid>> JoinNetwork<NI>
    for Reinforcement<NI, NO, N>
{
    type OutA = Sigmoid;
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO, OutA = Sigmoid>> ForwardNetwork<NI, NO>
    for Reinforcement<NI, NO, N>
{
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        let probabilities = self.network.forward(input);
        let ranges = self.output_ranges();
        let target: [Float; NO] = std::array::from_fn(|i| {
            let r = ranges[i].clone();
            let p = probabilities[i];
            if rand::random_range(r.clone()) > p {
                r.start().clone()
            } else {
                r.end().clone()
            }
        });
        self.network
            .update_gradient(self.r, target.sub(probabilities));
        target
    }
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO, OutA = Sigmoid>> Network<NI, NO>
    for Reinforcement<NI, NO, N>
{
    fn randomize(&mut self) {
        self.network.randomize();
    }
    fn update_gradient(&mut self, r: Float, delta: [Float; NO]) -> [Float; NI] {
        self.network.update_gradient(r, delta)
    }

    fn reset_gradient(&mut self) {
        self.network.reset_gradient();
    }

    fn apply_gradient(&mut self, alpha: Float) {
        self.network.apply_gradient(alpha);
    }
    fn rescale_gradient(&mut self, a: Float) {
        self.network.rescale_gradient(a);
    }
    fn norm2_gradient(&self) -> Float {
        self.network.norm2_gradient()
    }
    fn output_ranges(&self) -> [RangeInclusive<Float>; NO] {
        self.network.output_ranges()
    }
}
