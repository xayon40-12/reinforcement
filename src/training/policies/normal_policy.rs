use num::{Float, traits::FloatConst};
use rand_distr::{Distribution, StandardNormal};

use crate::training::{BackProp, Eval, Gradient, StochasticPolicy, Weights, mlp::MLP};

pub struct NormalPolicy<T: Float + FloatConst>
where
    StandardNormal: Distribution<T>,
{
    mlp: MLP<T>,
    sigma: T,
}

impl<T: Float + FloatConst> NormalPolicy<T>
where
    StandardNormal: Distribution<T>,
{
    pub fn new(mlp: MLP<T>, sigma: T) -> Self {
        NormalPolicy { mlp, sigma }
    }
    pub fn input_len(&self) -> usize {
        self.mlp.input_len()
    }
    pub fn output_len(&self) -> usize {
        self.mlp.output_len()
    }
    pub fn action<'a>(&self, state: &'a [T]) -> &'a [T] {
        &state[self.state_len() - self.output_len()..]
    }
}

impl<T: Float + FloatConst> Weights<T> for NormalPolicy<T>
where
    StandardNormal: Distribution<T>,
{
    fn weights_len(&self) -> usize {
        self.mlp.weights_len()
    }
}
impl<T: Float + FloatConst> Eval<T> for NormalPolicy<T>
where
    StandardNormal: Distribution<T>,
{
    fn state_len(&self) -> usize {
        self.mlp.state_len() + 2 * self.mlp.min_back_front_len() + self.output_len()
    }

    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        self.mlp
            .eval(input, weights, &mut state[..self.mlp.state_len()]);
    }
    fn input_len(&self) -> usize {
        self.mlp.input_len()
    }
    fn output_len(&self) -> usize {
        self.mlp.output_len()
    }
    fn output<'a>(&self, state: &'a [T]) -> &'a [T] {
        self.mlp.output(&state[..self.mlp.state_len()])
    }
    fn output_mut<'a>(&self, state: &'a mut [T]) -> &'a mut [T] {
        self.mlp.output_mut(&mut state[..self.mlp.state_len()])
    }
}
impl<T: Float + FloatConst> Gradient<T> for NormalPolicy<T>
where
    StandardNormal: Distribution<T>,
{
    fn compute_gradient(&self, input: &[T], weights: &[T], state: &mut [T], gradient: &mut [T]) {
        let probability = self.probability(state);
        let (state, tmp) = state.split_at_mut(self.mlp.state_len());
        let (back, tmp) = tmp.split_at_mut(self.mlp.min_back_front_len());
        let (front, action_state) = tmp.split_at_mut(self.mlp.min_back_front_len());
        back.iter_mut()
            .zip(self.mlp.output(state).iter())
            .zip(action_state.iter())
            .for_each(|((b, &m), &a)| *b = (a - m) / self.sigma.powi(2) * probability);
        self.mlp
            .back_prop(input, weights, state, front, back, gradient);
    }
}
impl<T: Float + FloatConst> StochasticPolicy<T> for NormalPolicy<T>
where
    StandardNormal: Distribution<T>,
{
    fn probability(&self, state: &[T]) -> T {
        self.output(state)
            .iter()
            .zip(self.action(state).iter())
            .fold(T::one(), |acc, (m, a)| acc * gaussian(*a, *m, self.sigma))
    }
    fn stochastic_eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        self.eval(input, weights, state);
        let (state, tmp) = state.split_at_mut(self.mlp.state_len());
        let (_back_front, action_state) = tmp.split_at_mut(2 * self.mlp.min_back_front_len());
        let means = self.mlp.output(state);
        let (min, max) = self.mlp.output_range();
        let limit = T::from(10.0).unwrap() * self.sigma; // NOTE: use to clamp to 10 sigma to avoid infinities
        action_state
            .iter_mut()
            .zip(means.iter())
            .for_each(|(a, &m)| {
                let min = min.unwrap_or(T::neg_infinity()).max(m - limit);
                let max = max.unwrap_or(T::infinity()).min(m + limit);
                *a = rand_distr::Normal::new(m, self.sigma)
                    .unwrap()
                    .sample(&mut rand::rng())
                    .clamp(min, max);
            });
    }
    fn stochastic_output<'a>(&self, state: &'a [T]) -> &'a [T] {
        &state[self.mlp.state_len()..]
    }
}
fn gaussian<T: Float + FloatConst>(x: T, mean: T, sigma: T) -> T {
    let two = T::one() + T::one();
    (-((x - mean) / sigma).powi(2) / two).exp() / (sigma * (T::PI() * two).sqrt())
}
