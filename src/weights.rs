use std::{iter::repeat, marker::PhantomData};

use num::{Float, traits::FloatConst};
use rand_distr::{Distribution, StandardNormal};

//FIXME: add proper debug_assert for all the methods here

pub trait Weights {
    fn weights_len(&self) -> usize;
    //TODO: add a randomize function
}
pub trait Eval<T: Float>: Weights {
    fn state_len(&self) -> usize;
    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]);
    fn output<'a>(&self, state: &'a [T]) -> &'a [T];
    fn empty_state(&self) -> Box<[T]> {
        vec![T::zero(); self.state_len()].into_boxed_slice()
    }
}
pub trait Gradient<T: Float>: Eval<T> {
    fn compute_gradient(&self, input: &[T], weights: &[T], state: &[T], gradient: &mut [T]);
    fn empty_gradient(&self) -> Box<[T]> {
        vec![T::zero(); self.weights_len()].into_boxed_slice()
    }
}
pub trait BackProp<T: Float>: Eval<T> {
    fn back_prop(
        &self,
        input: &[T],
        weights: &[T],
        state: &[T],
        front: &mut [T],
        back: &[T],
        gradient: &mut [T],
    );
}
/// It is `Gradient` with respect to the scalar probability
pub trait StochasticPolicy<T: Float>: Gradient<T> {
    fn probability(&self, state: &[T]) -> T;
}

pub trait Activation<T: Float>: BackProp<T> {
    fn range(&self) -> (Option<T>, Option<T>);
}

// ==================================================================

pub struct LayerMatrix<T: Float> {
    inputs: usize,
    outputs: usize,
    _phantom: PhantomData<T>,
}

impl<T: Float> LayerMatrix<T> {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        LayerMatrix {
            inputs,
            outputs,
            _phantom: PhantomData,
        }
    }
}

impl<T: Float> Weights for LayerMatrix<T> {
    fn weights_len(&self) -> usize {
        (self.inputs + 1) * self.outputs // +1 for the bias
    }
}
impl<T: Float> Eval<T> for LayerMatrix<T> {
    fn state_len(&self) -> usize {
        self.outputs
    }
    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        debug_assert!(
            input.len() == self.inputs
                && state.len() == self.outputs
                && weights.len() == self.weights_len()
        );
        state
            .iter_mut()
            .zip(weights.chunks(self.inputs + 1))
            .for_each(|(s, ws)| {
                *s = ws[1..]
                    .iter()
                    .zip(input.iter())
                    .fold(ws[0], |acc, (&w, &i)| acc + w * i)
            })
    }
    fn output<'a>(&self, state: &'a [T]) -> &'a [T] {
        debug_assert!(state.len() == self.outputs);
        state
    }
}
impl<T: Float> BackProp<T> for LayerMatrix<T> {
    fn back_prop(
        &self,
        input: &[T],
        weights: &[T],
        state: &[T],
        front: &mut [T],
        back: &[T],
        gradient: &mut [T],
    ) {
        debug_assert!(
            input.len() == self.inputs
                && weights.len() == self.weights_len()
                && state.len() == self.outputs
                && gradient.len() == self.weights_len()
                && front.len() == self.inputs
                && back.len() == self.outputs
        );
        front.iter_mut().for_each(|f| *f = T::zero());
        back.iter()
            .zip(weights.chunks(self.inputs + 1))
            .zip(gradient.chunks_mut(self.inputs + 1))
            .for_each(|((b, ws), gs)| {
                gs[0] = *b;
                front.iter_mut().zip(ws[1..].iter()).for_each(|(f, w)| {
                    *f = *f + *b * *w;
                });
                gs[1..].iter_mut().zip(input.iter()).for_each(|(g, i)| {
                    *g = *b * *i;
                });
            });
    }
}

// ==================================================================

pub struct ReLu {
    inputs: usize,
}
impl ReLu {
    pub fn new(inputs: usize) -> Self {
        ReLu { inputs }
    }
}
impl Weights for ReLu {
    fn weights_len(&self) -> usize {
        0
    }
}
impl<T: Float> Eval<T> for ReLu {
    fn state_len(&self) -> usize {
        self.inputs
    }
    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        debug_assert!(
            input.len() == self.inputs && weights.len() == 0 && state.len() == self.inputs
        );
        state
            .iter_mut()
            .zip(input.iter())
            .for_each(|(s, i)| *s = i.max(T::zero()));
    }
    fn output<'a>(&self, state: &'a [T]) -> &'a [T] {
        debug_assert!(state.len() == self.inputs);
        state
    }
}
impl<T: Float> BackProp<T> for ReLu {
    fn back_prop(
        &self,
        input: &[T],
        weights: &[T],
        state: &[T],
        front: &mut [T],
        back: &[T],
        gradient: &mut [T],
    ) {
        debug_assert!(
            input.len() == self.inputs
                && weights.len() == 0
                && state.len() == self.inputs
                && gradient.len() == 0
                && front.len() == self.inputs
                && back.len() == self.inputs
        );
        front
            .iter_mut()
            .zip(input.iter())
            .zip(back.iter())
            .for_each(|((f, i), b)| *f = *b * i.signum().max(T::zero()));
    }
}
impl<T: Float> Activation<T> for ReLu {
    fn range(&self) -> (Option<T>, Option<T>) {
        (Some(T::zero()), None)
    }
}

// ==================================================================

pub struct MLP<T: Float> {
    layers: Box<[(LayerMatrix<T>, Box<dyn Activation<T>>)]>,
}

impl<T: Float> MLP<T> {
    pub fn new(mut inputs: usize, layers: Vec<(usize, Box<dyn Activation<T>>)>) -> Self {
        MLP {
            layers: layers
                .into_iter()
                .map(|(outputs, activation)| {
                    let layer = (LayerMatrix::new(inputs, outputs), activation);
                    inputs = outputs;
                    layer
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
    pub fn output_range(&self) -> (Option<T>, Option<T>) {
        self.layers[self.layers.len() - 1].1.range()
    }
}

impl<T: Float> Weights for MLP<T> {
    fn weights_len(&self) -> usize {
        self.layers
            .iter()
            .map(|(l, a)| l.weights_len() + a.weights_len())
            .sum()
    }
}

impl<T: Float> Eval<T> for MLP<T> {
    fn state_len(&self) -> usize {
        self.layers
            .iter()
            .map(|(l, a)| l.state_len() + a.state_len())
            .sum()
    }

    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        debug_assert!(
            input.len() == self.layers[0].0.inputs
                && weights.len() == self.weights_len()
                && state.len() == self.state_len()
        );
        let mut input = &*input;
        let mut weights = &*weights;
        let mut state = &mut *state;
        for (l, a) in self.layers.iter() {
            l.eval(
                input,
                &weights[0..l.weights_len()],
                &mut state[0..l.state_len()],
            );
            weights = &weights[l.weights_len()..];
            (input, state) = state.split_at_mut(l.state_len());
            a.eval(
                input,
                &weights[0..a.weights_len()],
                &mut state[0..a.state_len()],
            );
            weights = &weights[a.weights_len()..];
            (input, state) = state.split_at_mut(a.state_len());
        }
    }
    fn output<'a>(&self, state: &'a [T]) -> &'a [T] {
        debug_assert!(state.len() == self.state_len());
        &state[self.state_len() - self.layers[self.layers.len() - 1].0.state_len()..]
    }
}
impl<T: Float> BackProp<T> for MLP<T> {
    fn back_prop(
        &self,
        input: &[T],
        weights: &[T],
        state: &[T],
        outer_front: &mut [T],
        back: &[T],
        gradient: &mut [T],
    ) {
        debug_assert!(
            input.len() == self.layers[0].0.inputs
                && weights.len() == self.weights_len()
                && state.len() == self.state_len()
                && gradient.len() == self.weights_len()
                && outer_front.len() == self.layers[0].0.inputs
                && back.len() == self.layers[self.layers.len() - 1].0.outputs
        );
        let mut gradients = Vec::with_capacity(self.layers.len());
        let mut weightss = Vec::with_capacity(self.layers.len());
        let mut gradient = &mut *gradient;
        let mut weights = &*weights;
        for len in self.layers.iter().map(|l| l.0.weights_len()) {
            let (a, b) = gradient.split_at_mut(len);
            gradients.push(a);
            gradient = b;
            let (a, b) = weights.split_at(len);
            weightss.push(a);
            weights = b;
        }
        let mut inputs = Vec::with_capacity(self.layers.len() * 2);
        inputs.push(input);
        let mut state = &*state;
        for l in self.layers.iter() {
            let (a, b) = state.split_at(l.0.state_len());
            inputs.push(a);
            if b.len() > 0 {
                let (c, d) = b.split_at(l.1.state_len());
                inputs.push(c);
                state = d;
            } else {
                state = b
            }
        }
        let mut back = back
            .iter()
            .cloned()
            .chain(repeat(T::zero()))
            .take(
                self.layers
                    .iter()
                    .fold(self.layers[0].0.inputs, |a, l| a.max(l.0.outputs)),
            )
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut front = back.clone();

        self.layers
            .iter()
            .zip(inputs.chunks(2))
            .zip(gradients.into_iter())
            .zip(weightss.into_iter())
            .for_each(|(((l, i), g), ws)| {
                l.1.back_prop(&i[1], &[], &[], &mut back, &front, g);
                l.1.back_prop(&i[0], ws, &i[1], &mut front, &back, g);
            });
        outer_front
            .iter_mut()
            .zip(front.iter())
            .for_each(|(of, f)| *of = *f);
    }
}

// ==================================================================

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
        self.mlp.layers[0].0.inputs
    }
    pub fn output_len(&self) -> usize {
        self.mlp.layers[0].0.outputs
    }
    pub fn action<'a>(&self, state: &'a [T]) -> &'a [T] {
        &state[self.state_len() - self.output_len()..]
    }
}

impl<T: Float + FloatConst> Weights for NormalPolicy<T>
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
        self.mlp.state_len() + self.output_len()
    }

    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        let (state, action_state) = state.split_at_mut(self.mlp.state_len());
        self.mlp.eval(input, weights, state);
        let means = self.mlp.output(state);
        let (min, max) = self.mlp.output_range();
        let nr = T::from(10.0).unwrap() * self.sigma; // NOTE: use to clamp to 10 sigma to avoid infinities
        action_state
            .iter_mut()
            .zip(means.iter())
            .for_each(|(a, &m)| {
                let min = min.unwrap_or(T::neg_infinity()).max(m - nr);
                let max = max.unwrap_or(T::infinity()).min(m + nr);
                *a = rand_distr::Normal::new(m, self.sigma)
                    .unwrap()
                    .sample(&mut rand::rng())
                    .clamp(min, max);
            });
    }

    fn output<'a>(&self, state: &'a [T]) -> &'a [T] {
        self.mlp.output(&state[..self.mlp.state_len()])
    }
}
impl<T: Float + FloatConst> Gradient<T> for NormalPolicy<T>
where
    StandardNormal: Distribution<T>,
{
    fn compute_gradient(&self, input: &[T], weights: &[T], state: &[T], gradient: &mut [T]) {
        let probability = self.probability(state);
        let back = self
            .output(state)
            .iter()
            .zip(self.action(state))
            .map(|(&m, &a)| (a - m) / self.sigma.powi(2) * probability)
            .collect::<Vec<_>>();
        self.mlp
            .back_prop(input, weights, state, &mut [], &back, gradient);
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
}
fn gaussian<T: Float + FloatConst>(x: T, mean: T, sigma: T) -> T {
    let two = T::one() + T::one();
    (-((x - mean) / sigma).powi(2) / two).exp() / (sigma * (T::PI() * two).sqrt())
}
