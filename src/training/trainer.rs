use std::{fmt::LowerExp, iter::Sum};

use num::Float;
use rand::Rng;
use rand_distr::uniform::{SampleBorrow, SampleUniform};

use super::{BackProp, Eval, Weights, mlp::MLP};

pub struct Trainer<T: Float> {
    mlp: MLP<T>,
    weights: Box<[T]>,
    state: Box<[T]>,
    back: Box<[T]>,
    front: Box<[T]>,
    gradient: Box<[T]>,
}
impl<T: Float> Trainer<T> {
    pub fn new(mlp: MLP<T>) -> Self {
        Self {
            weights: mlp.empty_weights(),
            state: mlp.empty_state(),
            back: vec![T::zero(); mlp.min_back_front_len()].into_boxed_slice(),
            front: vec![T::zero(); mlp.min_back_front_len()].into_boxed_slice(),
            gradient: mlp.empty_weights(),
            mlp,
        }
    }
    pub fn randomize_weights(&mut self, a: T)
    where
        T: SampleBorrow<T> + SampleUniform,
    {
        let mut rng = rand::rng();
        let uni = rand::distr::Uniform::new(-a, a).unwrap();
        self.weights.iter_mut().for_each(|w| *w = rng.sample(&uni));
    }
    pub fn train(&mut self, iterations: usize, alpha: T, training_data: &[(&[T], &[T])])
    where
        T: std::fmt::Debug + LowerExp + Sum,
    {
        let align = iterations.ilog10() as usize;
        for i in 0..iterations {
            let mut error = T::zero();
            for &(inputs, targets) in training_data.iter() {
                self.mlp.eval(inputs, &self.weights, &mut self.state);
                let outputs = self.mlp.output(&self.state);
                if cfg!(debug_assertions) {
                    if i % (iterations / 10) == 0 {
                        print!("{outputs:?} ");
                    }
                }
                self.back
                    .iter_mut()
                    .zip(targets.iter())
                    .zip(outputs.iter())
                    .for_each(|((b, &t), &o)| *b = t - o);
                error = error + self.back.iter().map(|&d| d * d).sum::<T>();
                self.mlp.back_prop(
                    inputs,
                    &self.weights,
                    &self.state,
                    &mut self.front,
                    &mut self.back,
                    &mut self.gradient,
                );
                self.weights
                    .iter_mut()
                    .zip(self.gradient.iter())
                    .for_each(|(w, g)| *w = *w + alpha * *g);
            }
            if cfg!(debug_assertions) {
                if i % (iterations / 10) == 0 {
                    println!("\n{i:<align$}: error = {:.1e}", error.sqrt());
                }
            }
        }
    }
}
