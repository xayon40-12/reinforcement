use std::ops::RangeInclusive;

use array_vector_space::ArrayVectorSpace;
use rand::Rng;

use super::{
    BoundedNetwork, Float, ForwardNetwork, JoinNetwork, Network,
    activation::{Activation, BoundedActivation},
};

#[derive(Clone)]
pub struct Layer<const NI: usize, const NO: usize, A: Activation> {
    weights: [([Float; NI], Float, A); NO],
    gradient: [[Float; NI]; NO],
    gradient_bias: [Float; NO],
    activations: [Float; NO],
    inputs: [Float; NI],
}

impl<const NI: usize, const NO: usize, A: Activation> Default for Layer<NI, NO, A> {
    fn default() -> Self {
        let mut s = Self {
            weights: [([0.0; NI], 0.0, A::default()); NO],
            gradient: [[0.0; NI]; NO],
            gradient_bias: [0.0; NO],
            activations: [0.0; NO],
            inputs: [0.0; NI],
        };
        s.randomize();
        s
    }
}

impl<const NI: usize, const NO: usize, A: Activation> JoinNetwork<NI> for Layer<NI, NO, A> {
    type OutA = A;
}
impl<const NI: usize, const NO: usize, A: Activation> ForwardNetwork<NI, NO> for Layer<NI, NO, A> {
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        self.inputs = input;
        std::array::from_fn(|o| {
            let (ws, bias, activation) = &self.weights[o];
            self.activations[o] = ws
                .iter()
                .zip(input.iter())
                .map(|(&a, &b)| a * b)
                .fold(*bias, |acc, x| acc + x);
            activation.apply(self.activations[o])
        })
    }
}
impl<const NI: usize, const NO: usize, A: Activation> Network<NI, NO> for Layer<NI, NO, A> {
    fn randomize(&mut self) {
        let mut rng = rand::rng();
        let a = 1.0 / NO as Float;
        let uni = rand::distr::Uniform::new(-a, a).unwrap();
        self.weights.iter_mut().for_each(|(ws, bias, _)| {
            ws.iter_mut().for_each(|w| *w = rng.sample(uni));
            *bias = rng.sample(uni);
        });
    }
    fn update_gradient(&mut self, relaxation: Float, deltas_in: [Float; NO]) -> [Float; NI] {
        let mut delta_out = [0.0; NI];
        for ((o, gs), gb) in (0..)
            .zip(self.gradient.iter_mut())
            .zip(self.gradient_bias.iter_mut())
        {
            let (ws, _bias, activation) = &self.weights[o];
            //TODO update bias with gb
            let delta = deltas_in[o] * activation.derivative(self.activations[o]);
            for (i, g) in (0..).zip(gs) {
                delta_out[i] += delta * ws[i];
                *g = (1.0 - relaxation) * *g + relaxation * delta * self.inputs[i];
            }
            *gb = (1.0 - relaxation) * *gb + relaxation * delta;
        }
        delta_out
    }
    fn reset_gradient(&mut self) {
        self.gradient = [[0.0; NI]; NO];
        self.gradient_bias = [0.0; NO];
    }
    fn apply_gradient(&mut self, alpha: Float) {
        self.weights
            .iter_mut()
            .zip(self.gradient.iter())
            .zip(self.gradient_bias.iter())
            .for_each(|(((ws, bias, _), gs), gb)| {
                ws.iter_mut()
                    .zip(gs.iter())
                    .for_each(|(w, g)| *w += alpha * *g);
                *bias += alpha * *gb;
            });
    }
    fn rescale_gradient(&mut self, a: Float) {
        self.gradient.scal_mul(a);
    }
    fn norm2_gradient(&self) -> Float {
        self.gradient.norm2()
    }
    fn add_gradient(&mut self, rhs: &Self) {
        self.gradient = self.gradient.add(rhs.gradient);
        self.gradient_bias = self.gradient_bias.add(rhs.gradient_bias);
    }
}
impl<const NI: usize, const NO: usize, A: BoundedActivation> BoundedNetwork<NI, NO>
    for Layer<NI, NO, A>
{
    fn output_ranges(&self) -> [RangeInclusive<Float>; NO] {
        self.weights.map(|(_, _, a)| a.range())
    }
}
