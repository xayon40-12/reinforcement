use std::ops::RangeInclusive;

use array_vector_space::ArrayVectorSpace;
use boxarray::boxarray;
use rand_distr::Distribution;

use super::{BoundedNetwork, Float, ForwardNetwork, JoinNetwork, Network};

#[derive(Copy, Clone, Default, Debug)]
pub struct Score {
    last: Float,
    diff: Float,
    mean: Float,
    var: Float,
}
impl Score {
    pub fn new(mean: Float) -> Score {
        Score {
            last: mean,
            diff: 0.0,
            mean,
            var: 1.0,
        }
    }

    pub fn update(&mut self, score: Float) -> Self {
        self.last = score;
        self.diff = score - self.mean;
        self.mean += 0.5 * self.diff;
        let a = 1e-4;
        self.var = (1.0 - a) * self.var + a * self.diff.powi(2);
        *self
    }

    pub fn sigma(&self) -> Float {
        self.var.sqrt()
    }

    pub fn last(&self) -> Float {
        self.last
    }
    pub fn mean(&self) -> Float {
        self.mean
    }
    pub fn diff(&self) -> Float {
        self.diff
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
        sigma: Float,
        tasks_ctx: &mut [(C, Score); NC],
        ctx_to_net: impl Fn(&C) -> [Float; NI],
        f: impl Fn(&mut C, [Float; NO]) -> (Float, bool)
        //WARNING: the output Float must be the score for only the current step and it should not depend on previous ones
        + Send
        + Sync,
    ) where
        Self: Sized + Send + Sync,
    {
        self.relaxation = relaxation;
        let mut nets: Box<[Self; NC]> = boxarray(self.clone());
        tasks_ctx
            .iter_mut()
            .zip(nets.iter_mut())
            .for_each(|((ctx, score), net)| {
                let mut total_score = 0.0;
                loop {
                    let (step_score, done) = f(ctx, net.normal_forward(ctx_to_net(ctx), sigma));
                    total_score += step_score;
                    if done {
                        let reward = score.update(total_score);
                        net.normalize_gradient();
                        net.rescale_gradient(reward.diff().atan());
                        break;
                    }
                }
            });
        nets.into_iter().for_each(|net| self.add_gradient(&net));
        self.apply_gradient(alpha / NC as Float); // NOTE: divide by the number of added gradients
        self.reset_gradient();
    }
    fn normal_forward(&mut self, input: [Float; NI], sigma: Float) -> [Float; NO] {
        let probabilities = self.network.forward(input);
        let ranges = self.output_ranges();
        let target: [Float; NO] = std::array::from_fn(|i| {
            let r = ranges[i].clone();
            let range = r.end() - *r.start();
            let p = probabilities[i];
            rand_distr::Normal::new(p, sigma / range)
                .unwrap()
                .sample(&mut rand::rng())
                .clamp(*r.start(), *r.end())
        });
        self.network
            .update_gradient(self.relaxation, target.sub(probabilities));
        target
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
