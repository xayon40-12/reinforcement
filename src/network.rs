pub type Float = f64;

pub trait Network<const NI: usize, const NO: usize> {
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO];
    /// The parameter $r$ is the ratio $0 < r <= 1$ for the relaxation of the update of the gradient. A value of $r = 1$ correspond to keep only the new value of the gradient, whereas $r = 0.5$ will average the new and last value.
    fn update_gradient(&mut self, r: Float, delta: [Float; NO]) -> [Float; NI];
    fn reset_gradient(&mut self);
    fn apply_gradient(&mut self, alpha: Float);
    fn train(
        &mut self,
        iterations: usize,
        alpha: Float,
        training_data: &[([Float; NI], [Float; NO])],
    ) {
        for i in 0..iterations {
            let mut error = 0.0;
            for &(inputs, targets) in training_data.iter() {
                let outputs = self.forward(inputs);
                if i % (iterations / 10) == 0 {
                    println!("{outputs:?}");
                }
                let delta = std::array::from_fn(|i| outputs[i] - targets[i]);
                error += delta.iter().map(|d| d * d).sum::<Float>();
                self.update_gradient(1.0, delta);
                self.apply_gradient(alpha);
            }
            if i % (iterations / 10) == 0 {
                println!("{i:>10}: {error}");
            }
        }
    }
}

#[derive(Copy, Clone)]
pub enum Activation {
    Relu,
    Sigmoid(Float),
}

impl Activation {
    fn apply(&self, input: Float) -> Float {
        use Activation::*;
        match self {
            Relu => input.max(0.0),
            Sigmoid(lambda) => (1.0 + (-input * lambda).exp()).recip(),
        }
    }
    fn derivative(&self, input: Float) -> Float {
        use Activation::*;
        match self {
            Relu => input.signum().max(0.0),
            Sigmoid(..) => {
                let sig = self.apply(input);
                sig * (1.0 - sig)
            }
        }
    }
}

pub struct Layer<const NI: usize, const NO: usize> {
    weights: [([Float; NI], Activation); NO],
    gradient: [[Float; NI]; NO],
    activations: [Float; NO],
    inputs: [Float; NI],
}

impl<const NI: usize, const NO: usize> Default for Layer<NI, NO> {
    fn default() -> Self {
        Self {
            weights: [([0.0; NI], Activation::Relu); NO],
            gradient: [[0.0; NI]; NO],
            activations: [0.0; NO],
            inputs: [0.0; NI],
        }
    }
}

impl<const NI: usize, const NO: usize> Network<NI, NO> for Layer<NI, NO> {
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        self.inputs = input;
        std::array::from_fn(|o| {
            let (ws, activation) = &self.weights[o];
            self.activations[o] = ws
                .iter()
                .zip(input.iter())
                .map(|(&a, &b)| a + b)
                .fold(0.0, |acc, x| acc + x);
            activation.apply(self.activations[o])
        })
    }
    fn update_gradient(&mut self, r: Float, deltas_in: [Float; NO]) -> [Float; NI] {
        let mut delta_out = [0.0; NI];
        for (o, gs) in (0..).zip(self.gradient.iter_mut()) {
            let (ws, activation) = &self.weights[o];
            let delta = deltas_in[o] * activation.derivative(self.activations[o]);
            for (i, g) in (0..).zip(gs) {
                delta_out[i] += delta * ws[i];
                *g = (1.0 - r) * *g + r * delta * self.inputs[i];
            }
        }
        delta_out
    }
    fn reset_gradient(&mut self) {
        self.gradient = [[0.0; NI]; NO];
    }
    fn apply_gradient(&mut self, alpha: Float) {
        self.weights
            .iter_mut()
            .zip(self.gradient.iter())
            .for_each(|((ws, _), gs)| {
                ws.iter_mut()
                    .zip(gs.iter())
                    .for_each(|(w, g)| *w -= alpha * *g)
            });
    }
}

#[derive(Default)]
pub struct Layers<const NI: usize, const NH: usize, const NO: usize, O: Network<NH, NO>> {
    layer_in: Layer<NI, NH>,
    layer_out: O,
}

impl<const NI: usize, const NH: usize, const NO: usize, O: Network<NH, NO>> Network<NI, NO>
    for Layers<NI, NH, NO, O>
{
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        self.layer_out.forward(self.layer_in.forward(input))
    }

    fn update_gradient(&mut self, r: Float, delta: [Float; NO]) -> [Float; NI] {
        self.layer_in
            .update_gradient(r, self.layer_out.update_gradient(r, delta))
    }
    fn reset_gradient(&mut self) {
        self.layer_in.reset_gradient();
        self.layer_out.reset_gradient();
    }
    fn apply_gradient(&mut self, alpha: Float) {
        self.layer_in.apply_gradient(alpha);
        self.layer_out.apply_gradient(alpha);
    }
}
