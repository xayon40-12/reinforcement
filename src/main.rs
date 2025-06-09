use reinforcement::network::{
    Float, Layer, Layers, Network, Reinforcement,
    activation::{Activation, Id, Relu, Sigmoid},
};

fn add() {
    //TODO use identity for activation for this case
    let mut net: Layers<2, 3, Id, Layer<3, 1, Id>> = Default::default();
    net.train(
        1000,
        1e-2,
        &[
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 1.0], [2.0]),
            ([1.0, 0.0], [1.0]),
            ([-1.0, -6.0], [-7.0]),
            ([-1.0, 6.0], [5.0]),
        ],
    );
}

fn xor() {
    let mut net: Layers<2, 2, Id, Layer<2, 1, Id>> = Default::default();
    net.train(
        100,
        1e-1,
        &[
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 1.0], [0.0]),
            ([1.0, 0.0], [1.0]),
        ],
    );
}

pub fn distance(a: [Float; 2], b: [Float; 2]) -> Float {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn reinforcement() {
    let mut net: Reinforcement<4, 2, Layer<4, 2, Sigmoid>> = Default::default();
    // let mut net: Reinforcement<4, 2, Layers<4, 10, Id, Layers<10, 10, Id, Layer<10, 2, Sigmoid>>>> =
    //     Default::default();

    let alpha = 1e-2;
    let targets = [[10.0, -7.0], [-40.0, -120.0], [300.0, 10.0], [-30.0, 100.0]];
    let start = [0.0; 2];
    let nb_reinforcement = 100000usize;

    let width = nb_reinforcement.ilog10() as usize;
    let mut best_distances = targets.map(|t| distance(start, t));
    for i in 0..nb_reinforcement {
        let mut poss = targets.map(|_| start);
        let mut d = targets.map(|_| 0.0);
        let max_ticks = 1 + i / 100;
        let alpha = alpha * (max_ticks as f64).recip();
        for ((((target, pos), d), best_distance), j) in targets
            .iter()
            .zip(poss.iter_mut())
            .zip(d.iter_mut())
            .zip(best_distances.iter_mut())
            .zip(0..)
        {
            let mut ticks = 0;
            net.reinforce(1e-5, |net| {
                let input = [pos[0], pos[1], target[0], target[1]];
                let output = net.forward(input);
                pos[0] += output[0];
                pos[1] += output[1];
                if j == 0 && i % (nb_reinforcement / 10) == 0 {
                    // println!("@ {} {}", pos[0], pos[1]);
                }
                *d = distance(*target, *pos);
                ticks += 1;
                if ticks > max_ticks {
                    if *d < *best_distance * 0.5 {
                        *best_distance = *d;
                        Some(alpha)
                    } else {
                        Some(-alpha)
                    }
                } else {
                    None
                }
            });
        }
        if i % (nb_reinforcement / 10) == 0 {
            println!("{i:<width$}: {d:?}");
        }
    }
}

fn main() {
    add();
    xor();
    reinforcement();
}
