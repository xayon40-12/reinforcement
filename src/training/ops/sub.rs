use std::marker::PhantomData;

use num::Float;

use crate::training::{Eval, Gradient, Weights};

pub struct Sub<T: Float, A: Gradient<T>, B: Gradient<T>> {
    a: A,
    b: B,
    _phantom: PhantomData<T>,
}
pub fn sub<T: Float, A: Gradient<T>, B: Gradient<T>>(a: A, b: B) -> Sub<T, A, B> {
    debug_assert!(
        a.weights_len() == b.weights_len() || a.weights_len() * b.weights_len() == 0,
        "weights len"
    );
    debug_assert!(
        a.output_len() == b.output_len() || a.output_len() == 1 || b.output_len() == 1,
        "output len"
    );
    Sub {
        a,
        b,
        _phantom: PhantomData,
    }
}

impl<T: Float, A: Gradient<T>, B: Gradient<T>> Weights<T> for Sub<T, A, B> {
    fn weights_len(&self) -> usize {
        self.a.weights_len() + self.b.weights_len()
    }
}
impl<T: Float, A: Gradient<T>, B: Gradient<T>> Eval<T> for Sub<T, A, B> {
    fn state_len(&self) -> usize {
        self.a.state_len() + self.b.state_len()
    }
    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        let (input_a, input_b) = input.split_at(self.a.input_len());
        let (weights_a, weights_b) = weights.split_at(self.a.weights_len());
        let (state_a, state_b) = state.split_at_mut(self.a.state_len());
        self.a.eval(input_a, weights_a, state_a);
        self.b.eval(input_b, weights_b, state_b);
        if self.a.output_len() == self.b.output_len() {
            self.a
                .output_mut(state_a)
                .iter_mut()
                .zip(self.b.output_mut(state_b).iter_mut())
                .for_each(|(a, b)| *a = *a - *b);
        } else {
            if self.a.output_len() == 1 {
                let a = self.a.output(state_a)[0];
                self.b
                    .output_mut(state_b)
                    .iter_mut()
                    .for_each(|b| *b = a - *b);
            } else {
                let b = self.b.output(state_b)[0];
                self.a
                    .output_mut(state_a)
                    .iter_mut()
                    .for_each(|a| *a = *a - b);
            }
        }
    }
    fn input_len(&self) -> usize {
        self.a.input_len() + self.b.input_len()
    }
    fn output_len(&self) -> usize {
        self.a.output_len().max(self.b.output_len())
    }
    fn output<'a>(&self, state: &'a [T]) -> &'a [T] {
        if self.a.output_len() < self.b.output_len() {
            self.b.output(&state[self.a.state_len()..])
        } else {
            self.a.output(&state[..self.a.state_len()])
        }
    }
    fn output_mut<'a>(&self, state: &'a mut [T]) -> &'a mut [T] {
        if self.a.output_len() < self.b.output_len() {
            self.b.output_mut(&mut state[self.a.state_len()..])
        } else {
            self.a.output_mut(&mut state[..self.a.state_len()])
        }
    }
}
impl<T: Float, A: Gradient<T>, B: Gradient<T>> Gradient<T> for Sub<T, A, B> {
    fn compute_gradient(&self, input: &[T], weights: &[T], state: &mut [T], gradient: &mut [T]) {
        let (input_a, input_b) = input.split_at(self.a.input_len());
        let (weights_a, weights_b) = weights.split_at(self.a.weights_len());
        let (state_a, state_b) = state.split_at_mut(self.a.state_len());
        let (gradient_a, gradient_b) = gradient.split_at_mut(self.a.weights_len());
        self.a
            .compute_gradient(input_a, weights_a, state_a, gradient_a);
        self.b
            .compute_gradient(input_b, weights_b, state_b, gradient_b);

        if gradient_a.len() == gradient_b.len() {
            gradient_a
                .iter_mut()
                .zip(gradient_b.iter_mut())
                .for_each(|(a, b)| {
                    *a = *a - *b;
                    *b = *a
                });
        } else {
            if gradient_a.len() == 1 {
                let a = gradient_a[0];
                gradient_b.iter_mut().for_each(|b| *b = a - *b);
            } else if gradient_b.len() == 1 {
                let b = gradient_b[0];
                gradient_a.iter_mut().for_each(|a| *a = *a - b);
            }
        }
    }
}
