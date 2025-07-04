use array_vector_space::ArrayVectorSpace;
use reinforcement::network::{
    Float, Network,
    activation::{Id, Relu, SigmoidSim},
    layer::Layer,
    layers::Layers,
    reinforcement::{Reinforcement, Reward},
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
    let mut net: Reinforcement<
        6,
        2,
        Layers<6, 10, Relu, Layers<10, 10, Relu, Layer<10, 2, SigmoidSim>>>,
    > = Default::default();
    net.randomize();

    let alpha = 1e-3;
    let relaxation = 1e-3;
    let shape = 30.0;

    const MAX_TICKS: usize = 200;
    let nb_reinforcement = 1000000usize;

    let targets = [
        [100.0 as Float, 100.0],
        [100.0, -100.0],
        [-100.0, -100.0],
        [-100.0, 100.0],
    ];
    let obstacle = 40.0;
    let start = [0.0; 2];
    let mut rewards = targets.map(|_| Reward::default());

    let width = nb_reinforcement.ilog10() as usize;
    println!("{:?}", targets.map(|[x, y]| (x * x + y * y).sqrt()));

    for i in 1..nb_reinforcement {
        let mut poss = targets.map(|_| (start, [0.0; 2]));
        let mut dtots = targets.map(|_| 0.0);
        let mut ds = targets.map(|_| 0.0);
        let mut trajectory = [[[0.0; 2]; 4]; MAX_TICKS];
        for (((((target, (pos, speed)), dtot), d), reward), j) in targets
            .iter()
            .zip(poss.iter_mut())
            .zip(dtots.iter_mut())
            .zip(ds.iter_mut())
            .zip(rewards.iter_mut())
            .zip(0..)
        {
            let mut ticks = 0;
            net.reinforce(relaxation, alpha, |net| {
                let input = [pos[0], pos[1], speed[0], speed[1], target[0], target[1]];
                let output = net.pert_forward(input, shape);
                *speed = speed.add(output.scal_mul(1e-1));
                let next_pos = pos.add(*speed);
                if (0..4)
                    .map(|i| {
                        next_pos.sub(targets[i].scal_mul(0.5)).norm2() < (obstacle as Float).powi(2)
                    })
                    .fold(false, |a, b| a || b)
                {
                    *speed = [0.0; 2];
                } else {
                    *pos = next_pos;
                }
                trajectory[ticks][j] = *pos;
                ticks += 1;
                *dtot += speed.norm2().sqrt();
                if ticks >= MAX_TICKS {
                    *d = distance(*target, *pos);
                    Some(reward.update(-*dtot / (1.0 + *d) * 1e-3 - *d))
                } else {
                    None
                }
            });
        }
        if i % (nb_reinforcement / 100) == 0 {
            std::fs::write(
                "tmp_trajectory.txt",
                trajectory
                    .map(|vs| vs.map(|v| v.map(|v| v.to_string()).join(" ")).join(" "))
                    .join("\n"),
            )
            .ok();
            println!(
                "{i:<width$} {MAX_TICKS:<3}: target dist {:.1} {:?} | tot dist {:.1} {:?}",
                ds.iter().fold(0.0 as Float, |a, v| a + v * v).sqrt(),
                ds.map(|v| v as i64),
                dtots.iter().fold(0.0 as Float, |a, v| a + v * v).sqrt(),
                dtots.map(|v| v as i64)
            );
        }
    }
}

fn main() {
    add();
    xor();
    reinforcement();
}
