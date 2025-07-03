use reinforcement::network::{
    Float, Layer, Layers, Network,
    activation::{Id, Sigmoid},
    reinforcement::Reinforcement,
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
    net.randomize();

    let alpha = 1e-5;
    let targets = [
        [10.0 as Float, -7.0],
        [-40.0, -120.0],
        [300.0, 10.0],
        [-30.0, 100.0],
    ];
    println!("{:?}", targets.map(|[x, y]| (x * x + y * y).sqrt()));
    let start = [0.0; 2];
    let nb_reinforcement = 1000000usize;

    let width = nb_reinforcement.ilog10() as usize;
    let init_distances = targets.map(|t| distance(start, t));
    let mut best_distances = init_distances;
    for i in 1..nb_reinforcement {
        let mut poss = targets.map(|_| start);
        let mut d = targets.map(|_| 0.0);
        let max_ticks = 350;
        let shape = 100.0;
        for ((((target, pos), d), best_distance), _init_distance) in targets
            .iter()
            .zip(poss.iter_mut())
            .zip(d.iter_mut())
            .zip(best_distances.iter_mut())
            .zip(init_distances)
        {
            let mut ticks = 0;
            net.reinforce(1e-3, |net| {
                let input = [pos[0], pos[1], target[0], target[1]];
                let output = net.pert_forward(input, shape);
                fn to_movement(p: Float) -> Float {
                    p * 2.0 - 1.0
                }
                pos[0] += to_movement(output[0]);
                pos[1] += to_movement(output[1]);
                ticks += 1;
                if ticks > max_ticks {
                    *d = distance(*target, *pos);
                    let diff = *best_distance - *d;
                    *best_distance -= 0.5 * diff;
                    Some(*best_distance * diff * alpha)
                } else {
                    None
                }
            });
        }
        if i % (nb_reinforcement / 100) == 0 {
            println!(
                "{i:<width$} {max_ticks:<3}: {:.1} {:?}",
                d.iter().fold(0.0 as Float, |a, v| a + v * v).sqrt(),
                d.map(|v| v as i64)
            );
        }
    }
}

fn main() {
    add();
    xor();
    reinforcement();
}
