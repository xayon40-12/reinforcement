use std::sync::Arc;

use array_vector_space::ArrayVectorSpace;
use egui::{Color32, Rect, Shape, Stroke, emath::RectTransform, epaint::PathStroke, pos2};

use crate::network::{
    Float,
    activation::{Id, Relu},
    layer::Layer,
    layers::Layers,
    reinforcement::{MetaParameters, Reinforcement},
};

use super::{Parameter, Simulation, Tag, UpadeParameter};

type Net = Reinforcement<
    6,
    2,
    Layers<6, 128, Relu, Layers<128, 128, Relu, Layer<128, 2, Id>>>,
    Layers<6, 128, Relu, Layers<128, 128, Relu, Layer<128, 1, Id>>>,
>;

#[derive(Clone, Copy)]
pub struct Circle {
    pos: [Float; 2],
    radius: Float,
}
#[derive(Clone)]
pub struct Ctx {
    target: Circle,
    obstacles: Arc<[Circle]>,
    pos: [Float; 2],
    speed: [Float; 2],
}

pub struct Acceleration {
    net: Net,
    obstacles: Arc<[Circle]>,
    targets: [Circle; 4],
    starts: [[Float; 2]; 4],
    poss: [[Float; 2]; 4],
    speeds: [[Float; 2]; 4],
    meta_parameters: MetaParameters,
    tot_reinforcement: usize,
}

impl Acceleration {
    pub fn new() -> Self {
        let meta_parameters = MetaParameters {
            alpha: 1e-1,
            alpha_score: 1e-1,
            relaxation: 1e-1,
            sigma: 1e1,
        };
        let targets = [
            Circle {
                pos: [100.0 as Float, 100.0],
                radius: 5.0,
            },
            Circle {
                pos: [100.0, -100.0],
                radius: 5.0,
            },
            Circle {
                pos: [-100.0, -100.0],
                radius: 5.0,
            },
            Circle {
                pos: [-100.0, 100.0],
                radius: 5.0,
            },
        ];
        let starts = std::array::from_fn(|i| {
            targets[(i + 1) % 4]
                .pos
                .add(targets[(i + 2) % 4].pos)
                .scal_mul(0.25)
        });
        let obstacles = Arc::new(std::array::from_fn::<_, 44, _>(|i| {
            if i < 4 {
                Circle {
                    pos: targets[i].pos.scal_mul(0.5),
                    radius: 40.0,
                }
            } else {
                let a = (i - 4) as Float / 40.0 * 3.1415 * 2.0;
                let r = 200.0;
                Circle {
                    pos: [a.cos() * r, a.sin() * r],
                    radius: 30.0,
                }
            }
        }));
        let mut s = Acceleration {
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
impl Simulation for Acceleration {
    type Tag = Param;
    type Ctx = Ctx;
    type ActionIn = [Float; 6];
    type ActionOut = [Float; 2];
    type PhysicsInfo = ();

    fn reset(&mut self) {
        self.tot_reinforcement = 0;
        self.net.randomize();
        self.poss = self.starts;
        self.speeds = [[0.0; 2]; 4];
    }

    fn meta_parameters(&self) -> MetaParameters {
        self.meta_parameters
    }
    fn egui_parameters(&self) -> Vec<Parameter<Param>> {
        use Param::*;
        vec![
            Parameter::Slider {
                tag: Alpha,
                value: self.meta_parameters.alpha,
                logarithmic: true,
                range: 1e-6..=1e0,
            },
            Parameter::Slider {
                tag: AlphaScore,
                value: self.meta_parameters.alpha_score,
                logarithmic: true,
                range: 1e-6..=1e0,
            },
            Parameter::Slider {
                tag: Relaxation,
                value: self.meta_parameters.relaxation,
                logarithmic: true,
                range: 1e-6..=1e0,
            },
            Parameter::Slider {
                tag: Sigma,
                value: self.meta_parameters.sigma,
                logarithmic: true,
                range: 1e-5..=1e3,
            },
        ]
    }
    fn update_parameter(&mut self, update: UpadeParameter<Self::Tag>) {
        match update {
            UpadeParameter::Slider { tag, value } => match tag {
                Param::Alpha => self.meta_parameters.alpha = value,
                Param::AlphaScore => self.meta_parameters.alpha_score = value,
                Param::Relaxation => self.meta_parameters.relaxation = value,
                Param::Sigma => self.meta_parameters.sigma = value,
            },
            _ => {}
        }
    }

    fn ctx_list(&self) -> Vec<Ctx> {
        (0..4)
            .map(|i| Ctx {
                target: self.targets[i],
                obstacles: Arc::clone(&self.obstacles),
                pos: self.poss[i],
                speed: self.speeds[i],
            })
            .collect()
    }
    fn max_physics_iteration() -> usize {
        100
    }

    fn score_goal(ctx: &Ctx, _info: ()) -> (Float, bool) {
        let mut score = 0.0;

        let d = ctx.target.pos.sub(ctx.pos).norm2().sqrt();
        // for Circle { pos, radius } in ctx.obstacles.iter() {
        //     let d = ctx.pos.sub(*pos).norm2().sqrt() - radius;
        //     score -= 1e1 / (1.0 + d).powi(3);
        // }
        // score -= d;
        let reached_target = d < ctx.target.radius;
        if reached_target {
            score += 1e2;
        } else {
            score -= d;
        }
        (score, reached_target)
    }
    fn physics(
        Ctx {
            obstacles,
            pos,
            speed,
            ..
        }: &mut Ctx,
        acceleration: [Float; 2],
    ) {
        let dt = 0.1;
        *speed = speed.add(acceleration.scal_mul(dt)); //.clamp(-1.0, 1.0));
        let next_pos = pos.add(speed.scal_mul(dt));
        let mut collision = false;
        for Circle { pos, radius } in obstacles.iter() {
            if next_pos.sub(*pos).norm2() < radius * radius {
                *speed = [0.0; 2];
                collision = true;
            }
        }
        if !collision {
            *pos = next_pos;
        }
    }
    fn ctx_to_action(ctx: &Ctx) -> [Float; 6] {
        let Ctx {
            target, pos, speed, ..
        } = ctx;
        [
            pos[0],
            pos[1],
            speed[0],
            speed[1],
            target.pos[0],
            target.pos[1],
        ]
    }
    fn reinforcement_network(
        &mut self,
    ) -> &mut impl crate::network::reinforcement::Reinforce<
        ActionIn = Self::ActionIn,
        ActionOut = Self::ActionOut,
    > {
        &mut self.net
    }

    fn render_rect(&self) -> Rect {
        Rect {
            min: pos2(-200.0, -200.0),
            max: pos2(200.0, 200.0),
        }
    }
    fn render(&self, to_screen: RectTransform, trajectories: Vec<Vec<Ctx>>) -> Vec<Shape> {
        let colors = [
            Color32::from_rgb(255, 0, 0),
            Color32::from_rgb(0, 255, 0),
            Color32::from_rgb(0, 0, 255),
            Color32::from_rgb(255, 192, 0),
        ];
        trajectories
            .iter()
            .zip(colors.into_iter())
            .map(|(t, c)| {
                Shape::line(
                    t.iter()
                        .map(|Ctx { pos: [x, y], .. }| to_screen * pos2(*x as _, *y as _))
                        .collect(),
                    PathStroke::new(5.0, c),
                )
            })
            .chain(self.obstacles.iter().map(
                |&Circle {
                     pos: [x, y],
                     radius,
                 }| {
                    Shape::circle_filled(
                        to_screen * pos2(x as _, y as _),
                        to_screen.scale().y * radius as f32,
                        Color32::GRAY,
                    )
                },
            ))
            .chain(self.targets.into_iter().zip(colors.into_iter()).map(
                |(
                    Circle {
                        pos: [x, y],
                        radius,
                    },
                    c,
                )| {
                    Shape::circle_stroke(
                        to_screen * pos2(x as _, y as _),
                        to_screen.scale().y * radius as f32,
                        Stroke::new(1.0, c),
                    )
                },
            ))
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
                    .iter()
                    .map(|t| &t[0])
                    .zip(colors.into_iter())
                    .map(|(Ctx { pos: [x, y], .. }, c)| {
                        Shape::circle_filled(
                            to_screen * pos2(*x as _, *y as _),
                            to_screen.scale().y * 3.0,
                            c,
                        )
                    }),
            )
            .collect()
    }
}
