use crate::training::{Direction, activations::id::Id, policies::normal_policy::NormalPolicy};

use super::{
    Eval, Optimizer, TimeStep, Value, Weights, least_squar_value::LeastSquareValue, mlp::MLP,
    optimizers::adam::Adam,
};

pub fn test_value_adam() {
    let mut value = LeastSquareValue::<f32>::new(MLP::new(
        2,
        vec![Id::layer(10), Id::layer(10), Id::layer(1)],
    ));
    let mut weights = value.empty_weights();
    weights
        .iter_mut()
        .for_each(|w| *w = rand::random_range(-1e-1f32..=1e-1));
    let mut gradient = value.empty_weights();
    let mut tmp_gradient = value.empty_weights();

    let mut adam = Adam::<f32>::new(value.weights_len()).with_alpha(1e-3);
    let f = |x, y| x + y;

    let n = 64;
    let coords = (0..64)
        .flat_map(|x| {
            (0..64).map(move |y| {
                [
                    2.0 * x as f32 / n as f32 - 1.0,
                    2.0 * y as f32 / n as f32 - 1.0,
                ]
            })
        })
        .collect::<Vec<_>>();
    let mut ctx = coords
        .iter()
        .cloned()
        .map(|input @ [x, y]| {
            let mut state = value.empty_state();
            value.set_target(f(x, y), &mut state);
            (input, state)
        })
        .collect::<Vec<_>>();
    let mut time_steps = ctx
        .iter_mut()
        .map(|(input, state)| TimeStep { input, state })
        .collect::<Vec<_>>();

    adam.optimize(
        10,
        64,
        &mut value,
        &mut weights,
        &mut gradient,
        &mut tmp_gradient,
        &mut time_steps,
        Direction::Descent,
    );

    let d = ctx
        .into_iter()
        .map(|(input @ [x, y], mut state)| {
            value.eval(&input, &weights, &mut state);
            (f(x, y) - value.value(&state)).powi(2)
        })
        .sum::<f32>()
        .sqrt();
    println!("d: {d:.2e}");
}

pub fn test_policy_adam() {
    let mut policy = NormalPolicy::<f32>::new(
        MLP::new(2, vec![Id::layer(10), Id::layer(10), Id::layer(1)]),
        0.1,
    );
    let mut weights = policy.empty_weights();
    weights
        .iter_mut()
        .for_each(|w| *w = rand::random_range(-1e-1f32..=1e-1));
    let mut gradient = policy.empty_weights();
    let mut tmp_gradient = policy.empty_weights();

    let mut adam = Adam::<f32>::new(policy.weights_len()).with_alpha(1e-3);
    let f = |x, y| x + y;

    let n = 64;
    let coords = (0..64)
        .flat_map(|x| {
            (0..64).map(move |y| {
                [
                    2.0 * x as f32 / n as f32 - 1.0,
                    2.0 * y as f32 / n as f32 - 1.0,
                ]
            })
        })
        .collect::<Vec<_>>();
    let mut ctx = coords
        .iter()
        .cloned()
        .map(|input| {
            let mut state = policy.empty_state();
            policy.eval(&input, &weights, &mut state);
            (input, state)
        })
        .collect::<Vec<_>>();
    let mut time_steps = ctx
        .iter_mut()
        .map(|(input, state)| TimeStep { input, state })
        .collect::<Vec<_>>();

    // FIXME: optimizing a stochastic policy by itself does not lead anywhere. An actual use of it such as PPO should be what is optimized. The following adam.optimize makes no sense, although it is still usefull as a test to check that there is no segmentatition fault.
    adam.optimize(
        10,
        64,
        &mut policy,
        &mut weights,
        &mut gradient,
        &mut tmp_gradient,
        &mut time_steps,
        Direction::Ascent,
    );

    let d = ctx
        .into_iter()
        .map(|(input @ [x, y], mut state)| {
            policy.eval(&input, &weights, &mut state);
            (f(x, y) - policy.output(&state)[0]).powi(2)
        })
        .sum::<f32>()
        .sqrt();
    println!("d: {d:.2e}");
}
