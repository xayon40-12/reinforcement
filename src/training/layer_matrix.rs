use std::marker::PhantomData;

use num::Float;

use super::{BackProp, Eval, Weights};

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
    pub fn inputs(&self) -> usize {
        self.inputs
    }
    pub fn outputs(&self) -> usize {
        self.outputs
    }
}

impl<T: Float> Weights<T> for LayerMatrix<T> {
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
        back: &mut [T],
        gradient: &mut [T],
    ) {
        debug_assert!(
            input.len() == self.inputs
                && weights.len() == self.weights_len()
                && state.len() == self.outputs
                && gradient.len() == self.weights_len()
                && front.len() == self.inputs
                && back.len() == self.outputs,
            "LayerMatrix"
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
