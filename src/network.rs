use activation::Activation;
use rand::Rng;

pub mod activation;
pub mod reinforcement;

pub type Float = f64;

pub trait JoinNetwork<const NI: usize> {}
pub trait ForwardNetwork<const NI: usize, const NO: usize>: JoinNetwork<NI> {
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO];
}
pub trait Network<const NI: usize, const NO: usize>: ForwardNetwork<NI, NO> {
    fn randomize(&mut self);
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
        let align = iterations.ilog10() as usize;
        for i in 0..iterations {
            let mut error = 0.0;
            for &(inputs, targets) in training_data.iter() {
                let outputs = self.forward(inputs);
                if i % (iterations / 10) == 0 {
                    print!("{outputs:?} ");
                }
                let normalization = (2.0 * NO as Float).recip();
                let delta = std::array::from_fn(|i| normalization * (targets[i] - outputs[i]));
                error += delta.iter().map(|d| d * d).sum::<Float>();
                self.update_gradient(1.0, delta);
                self.apply_gradient(alpha);
            }
            if i % (iterations / 10) == 0 {
                println!("\n{i:<align$}: error = {:.1e}", error.sqrt());
            }
        }
    }
}

pub struct Layer<const NI: usize, const NO: usize, A: Activation> {
    // TODO: put the activation function as a generic for the entire Layer
    weights: [([Float; NI], A); NO],
    gradient: [[Float; NI]; NO],
    activations: [Float; NO],
    inputs: [Float; NI],
}

impl<const NI: usize, const NO: usize, A: Activation> Default for Layer<NI, NO, A> {
    fn default() -> Self {
        let mut s = Self {
            weights: [([0.0; NI], A::default()); NO], // WARNING: this 1e-3 should be a random number
            gradient: [[0.0; NI]; NO],
            activations: [0.0; NO],
            inputs: [0.0; NI],
        };
        s.randomize();
        s
    }
}

impl<const NI: usize, const NO: usize, A: Activation> JoinNetwork<NI> for Layer<NI, NO, A> {}
impl<const NI: usize, const NO: usize, A: Activation> ForwardNetwork<NI, NO> for Layer<NI, NO, A> {
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        self.inputs = input;
        std::array::from_fn(|o| {
            let (ws, activation) = &self.weights[o];
            self.activations[o] = ws
                .iter()
                .zip(input.iter())
                .map(|(&a, &b)| a * b)
                .fold(0.0, |acc, x| acc + x);
            activation.apply(self.activations[o])
        })
    }
}
impl<const NI: usize, const NO: usize, A: Activation> Network<NI, NO> for Layer<NI, NO, A> {
    fn randomize(&mut self) {
        let mut rng = rand::rng();
        let a = 1e-1;
        let uni = rand::distr::Uniform::new(-a, a).unwrap();
        self.weights
            .iter_mut()
            .for_each(|(ws, _)| ws.iter_mut().for_each(|w| *w = rng.sample(uni)));
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
                    .for_each(|(w, g)| *w += alpha * *g)
            });
    }
}

#[derive(Default)]
pub struct Layers<const NI: usize, const NH: usize, A: Activation, O: JoinNetwork<NH>> {
    layer_in: Layer<NI, NH, A>,
    layer_out: O,
}
pub type LS<const NI: usize, const NH: usize, A, O> = Layers<NI, NH, A, O>;

impl<const NI: usize, const NH: usize, A: Activation, O: JoinNetwork<NH>> JoinNetwork<NI>
    for Layers<NI, NH, A, O>
{
}
impl<const NI: usize, const NH: usize, const NO: usize, A: Activation, O: Network<NH, NO>>
    ForwardNetwork<NI, NO> for Layers<NI, NH, A, O>
{
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        self.layer_out.forward(self.layer_in.forward(input))
    }
}
impl<const NI: usize, const NH: usize, const NO: usize, A: Activation, O: Network<NH, NO>>
    Network<NI, NO> for Layers<NI, NH, A, O>
{
    fn randomize(&mut self) {
        self.layer_in.randomize();
        self.layer_out.randomize();
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

#[derive(Default)]
pub struct Reinforcement<const NI: usize, const NO: usize, N: Network<NI, NO>> {
    network: N,
    r: Float,
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO>> Reinforcement<NI, NO, N> {
    pub fn reinforce(
        &mut self,
        r: Float,
        mut f: impl FnMut(&mut dyn ForwardNetwork<NI, NO>) -> Option<Float>,
    ) where
        Self: Sized,
    {
        self.r = r;
        loop {
            let alpha = f(self);
            if let Some(alpha) = alpha {
                self.apply_gradient(alpha);
                self.reset_gradient();
                break;
            }
        }
    }
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO>> JoinNetwork<NI>
    for Reinforcement<NI, NO, N>
{
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO>> ForwardNetwork<NI, NO>
    for Reinforcement<NI, NO, N>
{
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        let output = self.network.forward(input);
        self.network.update_gradient(self.r, output);
        output
    }
}
impl<const NI: usize, const NO: usize, N: Network<NI, NO>> Network<NI, NO>
    for Reinforcement<NI, NO, N>
{
    fn randomize(&mut self) {
        self.network.randomize();
    }
    fn update_gradient(&mut self, r: Float, delta: [Float; NO]) -> [Float; NI] {
        self.network.update_gradient(r, delta)
    }

    fn reset_gradient(&mut self) {
        self.network.reset_gradient();
    }

    fn apply_gradient(&mut self, alpha: Float) {
        self.network.apply_gradient(alpha);
    }
}
