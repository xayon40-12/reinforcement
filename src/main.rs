use reinforcement::network::{Float, Layer, Layers, Network, Reinforcement};

fn add() {
    //TODO use identity for activation for this case
    let mut net: Layers<2, 3, Layer<3, 1>> = Default::default();
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
    let mut net: Layers<2, 2, Layer<2, 1>> = Default::default();
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
    let mut net: Reinforcement<4, 2, Layers<4, 10, Layers<10, 10, Layer<10, 2>>>> =
        Default::default();

    let alpha = 1e-1;
    let target = [10.0, -7.0];
    let start = [0.0; 2];
    let max_ticks = 1000;
    let nb_reinforcement = 1000usize;

    let width = nb_reinforcement.ilog10() as usize;
    let mut best_distance = distance(start, target);
    for i in 0..nb_reinforcement {
        let mut pos = start;
        let mut ticks = 0;
        let mut d = 0.0;
        net.reinforce(1e-2, 1.0, |net| {
            let input = [pos[0], pos[1], target[0], target[1]];
            let output = net.forward(input);
            pos[0] += output[0];
            pos[1] += output[1];
            d = distance(target, pos);
            ticks += 1;
            if ticks > max_ticks {
                if d < best_distance {
                    best_distance = d;
                    Some(alpha)
                } else {
                    Some(-alpha)
                }
            } else {
                None
            }
        });
        if i % (nb_reinforcement / 10) == 0 {
            println!("{i:<width$}: {d:.2e}");
        }
    }
}

fn main() {
    add();
    xor();
    reinforcement();
}
