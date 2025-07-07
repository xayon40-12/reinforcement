use std::ops::RangeInclusive;

use array_vector_space::ArrayVectorSpace;
use rand_distr::Distribution;

use super::{
    BoundedNetwork, Float, ForwardNetwork, JoinNetwork, Network, StochasticForwardNetwork,
};

#[derive(Copy, Clone, Default, Debug)]
pub struct Reward {
    diff: Float,
    mean: Float,
    var: Float,
}
impl Reward {
    pub fn new(best: Float) -> Reward {
        Reward {
            diff: 0.0,
            mean: best,
            var: 1.0,
        }
    }

    pub fn update(&mut self, reward: Float) -> Float {
        self.diff = reward - self.mean;
        self.mean += 0.5 * self.diff;
        let a = 1e-4;
        self.var = (1.0 - a) * self.var + a * self.diff.powi(2);
        self.diff
    }

    pub fn sigma(&self) -> Float {
        self.var.sqrt()
    }
}

#[derive(Default, Clone)]
pub struct Reinforcement<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> {
    network: N,
    relaxation: Float,
}
impl<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> Reinforcement<NI, NO, N> {
    pub fn reinforce<const NC: usize, C: Send + Sync>(
        &mut self,
        relaxation: Float,
        alpha: Float,
        tasks_ctx: &mut [C; NC],
        f: impl Fn(
            &mut C,
            &mut dyn StochasticForwardNetwork<NI, NO, OutA = N::OutA>,
        ) -> Option<(Float, Float)>
        + Send
        + Sync,
    ) where
        Self: Sized + Send + Sync,
    {
        self.relaxation = relaxation;
        let mut nets: [_; NC] = std::array::from_fn(|_| self.clone());
        tasks_ctx
            .iter_mut()
            .zip(nets.iter_mut())
            .for_each(|(ctx, net)| {
                loop {
                    let reward_update = f(ctx, net);
                    if let Some((reward_update, global_scale)) = reward_update {
                        net.normalize_gradient();
                        net.rescale_gradient(reward_update.atan() * global_scale.clamp(0.0, 1.0));
                        break;
                    }
                }
            });
        nets.into_iter().for_each(|net| self.add_gradient(&net));
        self.apply_gradient(alpha / NC as Float); // NOTE: divide by the number of added gradients
        self.reset_gradient();
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
    fn add_gradient(&mut self, rhs: &Self) {
        self.network.add_gradient(&rhs.network);
    }
}
impl<const NI: usize, const NO: usize, N: BoundedNetwork<NI, NO>> BoundedNetwork<NI, NO>
    for Reinforcement<NI, NO, N>
{
    fn output_ranges(&self) -> [RangeInclusive<Float>; NO] {
        self.network.output_ranges()
    }
}
