use array_vector_space::ArrayVectorSpace;
use rand_distr::Distribution;

use crate::network::activation::Sigmoid;

use super::{
    Float, ForwardNetwork, JoinNetwork, Network,
    activation::{Activation, Id},
};

#[derive(Clone, Copy, Debug)]
pub struct MetaParameters {
    pub alpha: Float,
    pub alpha_score: Float,
    pub relaxation: Float,
    pub sigma: Float,
}

pub trait Reinforce {
    type ActionIn;
    type ActionOut;
    fn forward(&mut self, action_in: Self::ActionIn) -> Self::ActionOut;
    fn reinforce<C>(
        &mut self,
        meta_parameters: MetaParameters,
        ctx_list: &mut [C],
        ctx_to_action_in: impl Fn(&C) -> Self::ActionIn,
        max_iter: usize,
        physics_cost: impl Fn(&mut C, Self::ActionOut) -> (Float, bool), //WARNING: the output Float must be the score for only the current step and it should not depend on previous ones
    );
}

#[derive(Default, Clone)]
pub struct Reinforcement<
    const NI: usize,
    const NO: usize,
    N: Network<NI, NO>,
    S: Network<NI, 1, OutA = Id>,
> {
    network: N,
    score_network: S,
    relaxation: Float,
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO>, S: Network<NI, 1, OutA = Id>>
    Reinforcement<NI, NO, N, S>
{
    pub fn randomize(&mut self) {
        self.network.randomize();
        self.score_network.randomize();
    }
    fn normal_forward(&mut self, input: [Float; NI], sigma: Float) -> [Float; NO] {
        let probabilities = self.network.forward(input);
        let ranges = self.network.output_ranges();
        let target: [Float; NO] = std::array::from_fn(|i| {
            let (min, max) = ranges[i];
            let p = probabilities[i];
            let nr = 10.0 * sigma; // NOTE: use to clamp to 10 sigma to avoid infinities
            let min = min.unwrap_or(Float::NEG_INFINITY).max(p - nr);
            let max = max.unwrap_or(Float::INFINITY).min(p + nr);
            rand_distr::Normal::new(p, sigma)
                .unwrap()
                .sample(&mut rand::rng())
                .clamp(min, max)
        });
        self.network
            .update_gradient(self.relaxation, target.sub(probabilities));
        target
    }
}

impl<const NI: usize, const NO: usize, N: Network<NI, NO>, S: Network<NI, 1, OutA = Id>> Reinforce
    for Reinforcement<NI, NO, N, S>
{
    type ActionIn = [Float; NI];
    type ActionOut = [Float; NO];
    fn reinforce<C>(
        &mut self,
        meta_parameters: MetaParameters,
        ctx_list: &mut [C],
        ctx_to_action_in: impl Fn(&C) -> [Float; NI],
        max_iter: usize,
        physics_cost: impl Fn(&mut C, [Float; NO]) -> (Float, bool),
    ) {
        self.relaxation = meta_parameters.relaxation;

        // let mut nets: Box<[Self; NC]> = boxarray(self.clone()); //FIXME: this fails in wasm with "memory access out of bounds" and "Uncaught TypeError: Cannot read properties of null (reading 'querySelector')"
        let mut nets: Vec<Self> = ctx_list.iter().map(|_| self.clone()).collect();
        ctx_list
            .iter_mut()
            .zip(nets.iter_mut())
            .for_each(|(ctx, net)| {
                let mut total_score = 0.0;
                let [prediction_score] = net.score_network.forward(ctx_to_action_in(ctx));
                for _ in 0..max_iter {
                    let action = net.normal_forward(ctx_to_action_in(ctx), meta_parameters.sigma);
                    let (step_score, done) = physics_cost(ctx, action);
                    total_score += step_score;
                    if done {
                        break;
                    }
                }
                let reward = total_score - prediction_score;
                net.score_network.update_gradient(1.0, [reward]);
                net.score_network.normalize_gradient();
                net.network.normalize_gradient();
                net.network.rescale_gradient(Sigmoid.apply(reward));
            });
        nets.into_iter().for_each(|net| {
            self.network.add_gradient(&net.network);
            self.score_network.add_gradient(&net.score_network);
        });
        self.network
            .apply_gradient(meta_parameters.alpha / ctx_list.len() as Float);
        self.score_network
            .apply_gradient(meta_parameters.alpha_score / ctx_list.len() as Float);
        self.network.reset_gradient();
        self.score_network.reset_gradient();
    }
    fn forward(&mut self, action_in: Self::ActionIn) -> Self::ActionOut {
        self.network.forward(action_in)
    }
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO>, S: Network<NI, 1, OutA = Id>>
    JoinNetwork<NI> for Reinforcement<NI, NO, N, S>
{
    type OutA = N::OutA;
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO>, S: Network<NI, 1, OutA = Id>>
    ForwardNetwork<NI, NO> for Reinforcement<NI, NO, N, S>
{
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        self.network.forward(input)
    }
}
