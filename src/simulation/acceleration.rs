use array_vector_space::ArrayVectorSpace;
use boxarray::boxarray_;
use egui::{Color32, Shape, Stroke, emath::RectTransform, epaint::PathStroke, pos2};

use crate::network::{
    Float, ForwardNetwork,
    activation::{Id, Relu},
    layer::Layer,
    layers::Layers,
    reinforcement::{MetaParameters, Reinforcement},
};

use super::{Parameter, Reply, Request, Simulation, Tag, UpadeParameter};

const MAX_TICKS: usize = 100;
pub struct Sim {
    net: Reinforcement<
        6,
        2,
        Layers<6, 16, Relu, Layers<16, 16, Relu, Layer<16, 2, Id>>>,
        Layers<6, 16, Relu, Layers<16, 16, Relu, Layer<16, 1, Id>>>,
    >,
    obstacles: [([Float; 2], Float); 44],
    targets: [([Float; 2], Float); 4],
    starts: [[Float; 2]; 4],
    poss: [[Float; 2]; 4],
    speeds: [[Float; 2]; 4],
    meta_parameters: MetaParameters,
    tot_reinforcement: usize,
}
type Ctx<'a> = (
    ([Float; 2], Float),
    &'a [([Float; 2], Float)],
    [Float; 2],
    [Float; 2],
);
fn ctx<'a>(
    targets: &[([Float; 2], Float); 4],
    obstacles: &'a [([Float; 2], Float)],
    poss: &[[Float; 2]; 4],
    speeds: &[[Float; 2]; 4],
) -> Box<[Ctx<'a>; 4]> {
    boxarray_(|((), i)| (targets[i], obstacles, poss[i], speeds[i]))
}

fn physics((_, obstacles, pos, speed): &mut Ctx, acceleration: [Float; 2]) {
    let dt = 0.1;
    *speed = speed.add(acceleration.scal_mul(dt));
    let next_pos = pos.add(speed.scal_mul(dt));
    let mut collision = false;
    for (o, r) in obstacles.iter() {
        if next_pos.sub(*o).norm2() < r * r {
            *speed = [0.0; 2];
            collision = true;
        }
    }
    if !collision {
        *pos = next_pos;
    }
}
fn sim(ticks: usize, ctx: &mut Ctx, net_out: [Float; 2]) -> (Float, bool) {
    let mut score = 0.0;
    physics(ctx, net_out);
    let ((target, r), obstacles, pos, speed) = ctx;

    let d = target.sub(*pos).norm2().sqrt();
    for (o, r) in obstacles.iter() {
        let d = pos.sub(*o).norm2().sqrt() - r;
        score -= 1e1 / (1.0 + d).powi(3);
        if pos.sub(*o).norm2() < r * r {
            score -= 1e2 * speed.norm2();
        }
    }
    score -= d;
    if d < *r {
        score += 1e2;
    } else {
        score += 1e0;
    }
    (score, ticks >= MAX_TICKS || d < *r)
}
fn ctx_to_net(ctx: &Ctx) -> [Float; 6] {
    let ((target, _), _, pos, speed) = ctx;
    [pos[0], pos[1], speed[0], speed[1], target[0], target[1]]
}

impl Sim {
    pub fn new() -> Self {
        let meta_parameters = MetaParameters {
            alpha: 1e-2,
            alpha_score: 1e-1,
            relaxation: 1e-4,
            sigma: 1e1,
        };
        let targets = [
            ([100.0 as Float, 100.0], 5.0),
            ([100.0, -100.0], 5.0),
            ([-100.0, -100.0], 5.0),
            ([-100.0, 100.0], 5.0),
        ];
        let starts = std::array::from_fn(|i| {
            targets[(i + 1) % 4]
                .0
                .add(targets[(i + 2) % 4].0)
                .scal_mul(0.25)
        });
        let obstacles: [_; 44] = std::array::from_fn(|i| {
            if i < 4 {
                (targets[i].0.scal_mul(0.5), 40.0)
            } else {
                let a = (i - 4) as Float / 40.0 * 3.1415 * 2.0;
                let r = 200.0;
                ([a.cos() * r, a.sin() * r], 30.0)
            }
        });
        let mut s = Sim {
            net: Default::default(),
            obstacles,
            targets,
            starts,
            poss: starts,
            speeds: [[0.0; 2]; 4],
            meta_parameters,
            tot_reinforcement: 0,
        };
        s.net.randomize();
        s
    }
    pub fn reset(&mut self) {
        self.tot_reinforcement = 0;
        self.net.randomize();
        self.poss = self.starts;
        self.speeds = [[0.0; 2]; 4];
    }
    pub fn reinforce(&mut self) {
        let nb_reinforcement = 100;
        for _ in 0..nb_reinforcement {
            self.net.reinforce(
                self.meta_parameters,
                &mut ctx(
                    &self.targets,
                    self.obstacles.as_slice(),
                    &self.poss,
                    &self.speeds,
                ),
                ctx_to_net,
                sim,
            );
        }
        self.tot_reinforcement += nb_reinforcement;
    }
    pub fn simulate(
        &mut self,
        movement: bool,
    ) -> [([([Float; 2], [Float; 2]); MAX_TICKS], usize); 4] {
        let mut trajectories = [([([0.0; 2], [0.0; 2]); MAX_TICKS], 0); 4];
        if movement {
            self.poss = self.starts;
            self.speeds = [[0.0; 2]; 4];
        }
        let mut ctxs = ctx(
            &self.targets,
            self.obstacles.as_slice(),
            &self.poss,
            &self.speeds,
        );
        for (ctx, (trajectory, ticks)) in ctxs.iter_mut().zip(trajectories.iter_mut()) {
            *ticks = MAX_TICKS;
            for i in 0..MAX_TICKS {
                let done = sim(i, ctx, self.net.forward(ctx_to_net(ctx))).1;
                let (_, _, pos, speed) = ctx;
                trajectory[i] = (*pos, *speed);
                if done {
                    *ticks = i;
                    break;
                }
            }
        }
        for (i, ((t, r), ..)) in (0..).zip(ctxs.into_iter()) {
            let (n_pos, n_speed) = trajectories[i].0[0];
            if n_pos.sub(t).norm2() < r * r {
                self.poss[i] = self.starts[i];
                self.speeds[i] = [0.0; 2];
            } else if movement {
                self.poss[i] = n_pos;
                self.speeds[i] = n_speed;
            }
        }
        trajectories
    }
    pub fn shapes(
        &self,
        to_screen: RectTransform,
        trajectories: [([([Float; 2], [Float; 2]); MAX_TICKS], usize); 4],
    ) -> Vec<Shape> {
        let colors = [
            Color32::from_rgb(255, 0, 0),
            Color32::from_rgb(0, 255, 0),
            Color32::from_rgb(0, 0, 255),
            Color32::from_rgb(255, 192, 0),
        ];
        trajectories
            .iter()
            .zip(colors.into_iter())
            .map(|((t, n), c)| {
                Shape::line(
                    t.iter()
                        .take(*n)
                        .map(|&([x, y], _)| to_screen * pos2(x as _, y as _))
                        .collect(),
                    PathStroke::new(5.0, c),
                )
            })
            .chain(self.obstacles.into_iter().map(|([x, y], r)| {
                Shape::circle_filled(
                    to_screen * pos2(x as _, y as _),
                    to_screen.scale().y * r as f32,
                    Color32::GRAY,
                )
            }))
            .chain(
                self.targets
                    .into_iter()
                    .zip(colors.into_iter())
                    .map(|(([x, y], r), c)| {
                        Shape::circle_stroke(
                            to_screen * pos2(x as _, y as _),
                            to_screen.scale().y * r as f32,
                            Stroke::new(1.0, c),
                        )
                    }),
            )
            .chain(
                self.starts
                    .into_iter()
                    .zip(colors.into_iter())
                    .map(|([x, y], c)| {
                        Shape::circle_filled(
                            to_screen * pos2(x as _, y as _),
                            to_screen.scale().y * 5.0,
                            c,
                        )
                    }),
            )
            .chain(
                trajectories
                    .into_iter()
                    .map(|(t, _)| t[0])
                    .zip(colors.into_iter())
                    .map(|(([x, y], _), c)| {
                        Shape::circle_filled(
                            to_screen * pos2(x as _, y as _),
                            to_screen.scale().y * 3.0,
                            c,
                        )
                    }),
            )
            .collect()
    }
}

#[derive(Clone, Copy)]
pub enum Param {
    Alpha,
    AlphaScore,
    Relaxation,
    Sigma,
}
impl Tag for Param {
    fn str(&self) -> &'static str {
        match self {
            Param::Alpha => "alpha",
            Param::AlphaScore => "alpha_score",
            Param::Relaxation => "relaxation",
            Param::Sigma => "sigma",
        }
    }
}
impl Simulation<4, Param> for Sim {
    fn handle_request(&mut self, request: Request<Param>) -> Option<Reply> {
        match request {
            Request::UpdateParameter(UpadeParameter { tag, value }) => {
                match tag {
                    Param::Alpha => self.meta_parameters.alpha = value,
                    Param::AlphaScore => self.meta_parameters.alpha_score = value,
                    Param::Relaxation => self.meta_parameters.relaxation = value,
                    Param::Sigma => self.meta_parameters.sigma = value,
                };
            }
            Request::Reinforce => self.reinforce(),
            Request::Reset => self.reset(),
            Request::Render(to_screen) => {
                let trajectories = self.simulate(false);
                return Some(Reply::Shapes(self.shapes(to_screen, trajectories)));
            }
        }
        None
    }

    fn parameters(&self) -> [Parameter<Param>; 4] {
        use Param::*;
        [
            Parameter {
                tag: Alpha,
                value: self.meta_parameters.alpha,
                logarithmic: true,
                range: 1e-6..=1e0,
            },
            Parameter {
                tag: AlphaScore,
                value: self.meta_parameters.alpha_score,
                logarithmic: true,
                range: 1e-6..=1e0,
            },
            Parameter {
                tag: Relaxation,
                value: self.meta_parameters.relaxation,
                logarithmic: true,
                range: 1e-6..=1e0,
            },
            Parameter {
                tag: Sigma,
                value: self.meta_parameters.sigma,
                logarithmic: true,
                range: 1e-5..=1e3,
            },
        ]
    }
}
