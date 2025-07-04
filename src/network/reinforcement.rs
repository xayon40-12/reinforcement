use std::ops::RangeInclusive;

use array_vector_space::ArrayVectorSpace;
use rand_distr::Distribution;

use super::{
    BoundedNetwork, Float, ForwardNetwork, JoinNetwork, Network, StochasticForwardNetwork,
};

#[derive(Copy, Clone, Default, Debug)]
pub struct Reward {
    diff: Float,
    best: Float,
}
impl Reward {
    pub fn new(best: Float) -> Reward {
        Reward { diff: 0.0, best }
    }

    pub fn update(&mut self, reward: Float) -> Float {
        self.diff = reward - self.best;
        self.best += 0.5 * self.diff;
        self.diff // TODO: should consider a way to reduce the return value as the best reward becomes better (this needs to know what was the initial reward)
    }
}

#[derive(Default)]
pub struct Reinforcement<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> {
    network: N,
    relaxation: Float,
}
impl<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> Reinforcement<NI, NO, N> {
    pub fn reinforce(
        &mut self,
        relaxation: Float,
        alpha: Float,
        mut f: impl FnMut(&mut dyn StochasticForwardNetwork<NI, NO, OutA = N::OutA>) -> Option<Float>,
    ) where
        Self: Sized,
    {
        self.relaxation = relaxation;
        loop {
            let reward_update = f(self);
            if let Some(reward_update) = reward_update {
                self.normalize_gradient();
                self.apply_gradient(reward_update * alpha);
                self.reset_gradient();
                break;
            }
        }
    }
}
impl<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> JoinNetwork<NI>
    for Reinforcement<NI, NO, N>
{
    type OutA = N::OutA;
}
impl<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> ForwardNetwork<NI, NO>
    for Reinforcement<NI, NO, N>
{
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        self.network.forward(input)
    }
}
impl<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> StochasticForwardNetwork<NI, NO>
    for Reinforcement<NI, NO, N>
{
    fn pert_forward(&mut self, input: [Float; NI], shape: Float) -> [Float; NO] {
        let probabilities = self.network.forward(input);
        let ranges = self.output_ranges();
        let target: [Float; NO] = std::array::from_fn(|i| {
            let r = ranges[i].clone();
            let p = probabilities[i];
            rand_distr::Pert::new(*r.start(), *r.end())
                .with_shape(shape)
                .with_mode(p)
                .unwrap()
                .sample(&mut rand::rng())
        });
        self.network
            .update_gradient(self.relaxation, target.sub(probabilities));
        target
    }
}
impl<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> Network<NI, NO>
    for Reinforcement<NI, NO, N>
{
    fn randomize(&mut self) {
        self.network.randomize();
    }
    fn update_gradient(&mut self, relaxation: Float, delta: [Float; NO]) -> [Float; NI] {
        self.network.update_gradient(relaxation, delta)
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
}
impl<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> BoundedNetwork<NI, NO>
    for Reinforcement<NI, NO, N>
{
    fn output_ranges(&self) -> [RangeInclusive<Float>; NO] {
        self.network.output_ranges()
    }
}
