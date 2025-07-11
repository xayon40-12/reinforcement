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
    fn reinforce<C: Clone>(
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
        self.network.update_gradient(1.0, target.sub(probabilities));
        target
    }
}

impl<const NI: usize, const NO: usize, N: Network<NI, NO>, S: Network<NI, 1, OutA = Id>> Reinforce
    for Reinforcement<NI, NO, N, S>
{
    type ActionIn = [Float; NI];
    type ActionOut = [Float; NO];
    fn reinforce<C: Clone>(
        &mut self,
        meta_parameters: MetaParameters,
        ctx_list: &mut [C],
        ctx_to_action_in: impl Fn(&C) -> [Float; NI],
        max_iter: usize,
        physics_cost: impl Fn(&mut C, [Float; NO]) -> (Float, bool),
    ) {
        self.relaxation = meta_parameters.relaxation;

        let clone_self = {
            let mut reinforcement = self.clone();
            reinforcement.network.reset_gradient();
            move || reinforcement.clone()
        };
        ctx_list
            .iter_mut()
            .map(|ctx| {
                let mut evolution = Vec::with_capacity(max_iter);
                for _ in 0..max_iter {
                    let mut reinforcement = clone_self();
                    let action_in = ctx_to_action_in(ctx);
                    let action = reinforcement.normal_forward(action_in, meta_parameters.sigma);
                    let (step_score, done) = physics_cost(ctx, action);
                    evolution.push((step_score, reinforcement, action_in));
                    if done {
                        break;
                    }
                }
                let (final_rein, _) = evolution.into_iter().rev().fold(
                    (clone_self(), 0.0),
                    |(mut final_rein, mut discounted_score),
                     (step_score, mut reinforcement, action_in)| {
                        discounted_score =
                            meta_parameters.relaxation * discounted_score + step_score;

                        let [prediciton] = reinforcement.score_network.forward(action_in);
                        let reward = discounted_score - prediciton;

                        reinforcement
                            .network
                            .rescale_gradient(Sigmoid.apply(reward));
                        reinforcement.score_network.update_gradient(1.0, [reward]);

                        final_rein.network.add_gradient(&reinforcement.network);
                        final_rein
                            .score_network
                            .add_gradient(&reinforcement.score_network);

                        (final_rein, discounted_score)
                    },
                );
                final_rein
            })
            .for_each(|mut cumulated_rein| {
                cumulated_rein.network.normalize_gradient();
                cumulated_rein.score_network.normalize_gradient();
                self.network.add_gradient(&cumulated_rein.network);
                self.score_network
                    .add_gradient(&cumulated_rein.score_network);
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
