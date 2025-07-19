use num::Float;

use crate::training::Optimizer;

pub struct Adam<T: Float> {
    alpha: T,
    beta_1: T,
    beta_2: T,
    epsilon: T,
    cumulated_beta_1: T,
    cumulated_beta_2: T,
    moment_1: Box<[T]>,
    moment_2: Box<[T]>,
}

impl<T: Float> Adam<T> {
    pub fn new(weights_len: usize) -> Self {
        Adam {
            alpha: T::from(0.001).unwrap(),
            beta_1: T::from(0.9).unwrap(),
            beta_2: T::from(0.999).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
            cumulated_beta_1: T::one(),
            cumulated_beta_2: T::one(),
            moment_1: vec![T::zero(); weights_len].into_boxed_slice(),
            moment_2: vec![T::zero(); weights_len].into_boxed_slice(),
        }
    }
    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }
    pub fn with_parameters(mut self, alpha: T, beta_1: T, beta_2: T, epsilon: T) -> Self {
        self.alpha = alpha;
        self.beta_1 = beta_1;
        self.beta_2 = beta_2;
        self.epsilon = epsilon;
        self
    }
}

impl<T: Float + std::fmt::Display> Optimizer<T> for Adam<T> {
    fn step(&mut self, weights: &mut [T], gradient: &mut [T]) {
        self.cumulated_beta_1 = self.cumulated_beta_1 * self.beta_1;
        self.cumulated_beta_2 = self.cumulated_beta_2 * self.beta_2;
        self.moment_1
            .iter_mut()
            .zip(gradient.iter())
            .for_each(|(m1, g)| *m1 = self.beta_1 * *m1 + (T::one() - self.beta_1) * *g);
        self.moment_2
            .iter_mut()
            .zip(gradient.iter())
            .for_each(|(m2, g)| *m2 = self.beta_2 * *m2 + (T::one() - self.beta_2) * (*g * *g));
        let c1 = T::one() - self.cumulated_beta_1;
        let c2 = (T::one() - self.cumulated_beta_2).sqrt();
        let alpha_step = self.alpha * c2 / c1;
        let epsilon_step = self.epsilon * c2;
        weights
            .iter_mut()
            .zip(self.moment_1.iter().zip(self.moment_2.iter()))
            .for_each(|(w, (&m1, &m2))| *w = *w - alpha_step * m1 / (m2.sqrt() + epsilon_step));
    }
}
