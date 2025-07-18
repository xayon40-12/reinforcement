use array_vector_space::{ArrayVectorSpace, ArrayVectorSpaceMut};
use boxarray::boxarray;
use rand::Rng;

use super::{Float, ForwardNetwork, JoinNetwork, Network, activation::Activation};

#[derive(Clone)]
pub struct Layer<const NI: usize, const NO: usize, A: Activation> {
    weights: Box<[([Float; NI], Float, A); NO]>,
    gradient: Box<[[Float; NI]; NO]>,
    gradient_bias: Box<[Float; NO]>,
    activations: Box<[Float; NO]>,
    inputs: Box<[Float; NI]>,
}

impl<const NI: usize, const NO: usize, A: Activation> Default for Layer<NI, NO, A> {
    fn default() -> Self {
        let mut s = Self {
            weights: boxarray(([0.0; NI], 0.0, A::default())),
            gradient: boxarray([0.0; NI]),
            gradient_bias: boxarray(0.0),
            activations: boxarray(0.0),
            inputs: boxarray(0.0),
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
        *self.inputs = input;
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
        self.gradient = boxarray([0.0; NI]);
        self.gradient_bias = boxarray(0.0);
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
        self.gradient.mut_scal_mul(a);
        self.gradient_bias.mut_scal_mul(a);
    }
    fn norm2_gradient(&self) -> Float {
        self.gradient.norm2()
    }
    fn add_gradient(&mut self, rhs: &Self) {
        self.gradient.mut_add(&rhs.gradient);
        self.gradient_bias.mut_add(&rhs.gradient_bias);
    }
    fn output_ranges(&self) -> [(Option<Float>, Option<Float>); NO] {
        self.weights.map(|(_, _, a)| a.range())
    }
}
