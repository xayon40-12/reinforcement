use std::time::Duration;

use array_vector_space::ArrayVectorSpace;
use iced::widget::{column, container, iced, slider, text, vertical_slider};
use iced::{Center, Element, Fill, Subscription, time};
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

pub fn main() -> iced::Result {
    add();
    xor();

    iced::application("Test", Slider::update, Slider::view)
        .subscription(Slider::subscription)
        .run()
}

#[derive(Debug, Clone)]
pub enum Message {
    SliderChanged(u8),
    Tick,
}

pub struct Slider {
    value: u8,
    net: Reinforcement<6, 2, Layers<6, 10, Relu, Layers<10, 10, Relu, Layer<10, 2, SigmoidSim>>>>,
    alpha: Float,
    relaxation: Float,
    shape: Float,
    obstacles_radius: Float,
    targets: [[Float; 2]; 4],
    rewards: [Reward; 4],
    start: [Float; 2],
    nb_reinforcement: usize,
    ds: [Float; 4],
    dtots: [Float; 4],
    trajectory: [[[Float; 2]; 4]; MAX_TICKS],
}

const MAX_TICKS: usize = 200;
impl Slider {
    fn new() -> Self {
        let mut s = Slider {
            value: 50,
            net: Default::default(),
            alpha: 1e-3,
            relaxation: 1e-3,
            shape: 30.0,
            obstacles_radius: 40.0,
            targets: [
                [100.0 as Float, 100.0],
                [100.0, -100.0],
                [-100.0, -100.0],
                [-100.0, 100.0],
            ],
            start: [0.0; 2],
            rewards: [Reward::default(); 4],
            nb_reinforcement: 50,
            ds: [0.0; 4],
            dtots: [0.0; 4],
            trajectory: [[[0.0; 2]; 4]; MAX_TICKS],
        };
        s.net.randomize();
        s
    }
    fn subscription(&self) -> Subscription<Message> {
        // 16 ms ≈ 60 Hz
        time::every(Duration::from_millis(16)).map(|_| Message::Tick)
    }

    fn update(&mut self, message: Message) {
        match message {
            Message::SliderChanged(value) => {
                self.value = value;
            }
            Message::Tick => {
                self.reinforce();
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let h_slider = container(
            slider(1..=100, self.value, Message::SliderChanged)
                .default(50)
                .shift_step(5),
        )
        .width(250);

        let v_slider = container(
            vertical_slider(1..=100, self.value, Message::SliderChanged)
                .default(50)
                .shift_step(5),
        )
        .height(200);

        let text = text(format!(
            "{MAX_TICKS:<3}: target dist {:.1} {:?} | tot dist {:.1} {:?}",
            self.ds.iter().fold(0.0 as Float, |a, v| a + v * v).sqrt(),
            self.ds.map(|v| v as i64),
            self.dtots
                .iter()
                .fold(0.0 as Float, |a, v| a + v * v)
                .sqrt(),
            self.dtots.map(|v| v as i64)
        ));

        column![v_slider, h_slider, text, iced(self.value as f32),]
            .width(Fill)
            .align_x(Center)
            .spacing(20)
            .padding(20)
            .into()
    }

    fn reinforce(&mut self) {
        for _ in 0..self.nb_reinforcement {
            let mut poss = self.targets.map(|_| (self.start, [0.0; 2]));
            self.dtots = self.targets.map(|_| 0.0);
            self.ds = self.targets.map(|_| 0.0);
            for (((((target, (pos, speed)), dtot), d), reward), j) in self
                .targets
                .iter()
                .zip(poss.iter_mut())
                .zip(self.dtots.iter_mut())
                .zip(self.ds.iter_mut())
                .zip(self.rewards.iter_mut())
                .zip(0..)
            {
                let mut ticks = 0;
                self.net.reinforce(self.relaxation, self.alpha, |net| {
                    let input = [pos[0], pos[1], speed[0], speed[1], target[0], target[1]];
                    let output = net.pert_forward(input, self.shape);
                    *speed = speed.add(output.scal_mul(1e-1));
                    let next_pos = pos.add(*speed);
                    if (0..4)
                        .map(|i| {
                            next_pos.sub(self.targets[i].scal_mul(0.5)).norm2()
                                < (self.obstacles_radius as Float).powi(2)
                        })
                        .fold(false, |a, b| a || b)
                    {
                        *speed = [0.0; 2];
                    } else {
                        *pos = next_pos;
                    }
                    self.trajectory[ticks][j] = *pos;
                    ticks += 1;
                    *dtot += speed.norm2().sqrt();
                    if ticks >= MAX_TICKS {
                        *d = target.sub(*pos).norm2().sqrt();
                        Some(reward.update(-*dtot / (1.0 + *d) * 1e-3 - *d))
                    } else {
                        None
                    }
                });
            }
        }
    }
}

impl Default for Slider {
    fn default() -> Self {
        Self::new()
    }
}
