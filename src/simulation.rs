use std::{ops::RangeInclusive, usize};

use egui::{Frame, Rect, Shape, emath::RectTransform};
use futures::channel::mpsc::{Receiver, Sender, channel};

use crate::network::Float;

pub mod acceleration;

pub trait Tag: Copy + Send + 'static {
    fn str(&self) -> &'static str;
}

pub enum Request<T> {
    UpdateParameter(UpadeParameter<T>),
    Reinforce,
    Reset,
    Render(RectTransform),
}

pub enum Reply {
    Shapes(Vec<Shape>),
}

pub struct Parameter<T> {
    pub tag: T,
    pub value: Float,
    pub logarithmic: bool,
    pub range: RangeInclusive<Float>,
}

pub struct UpadeParameter<T> {
    pub tag: T,
    pub value: Float,
}

pub trait Simulation<const NP: usize, T: Tag>: Send + 'static {
    fn parameters(&self) -> [Parameter<T>; NP];
    fn handle_request(&mut self, request: Request<T>) -> Option<Reply>;
}

pub struct SimulationGUI<const NP: usize, T: Tag, S: Simulation<NP, T>, FS: Fn() -> S> {
    parameters: [Parameter<T>; NP],
    simulation_request_tx: Sender<Request<T>>,
    simulation_reply_rx: Receiver<Reply>,
    create_simulation: FS,
    cached_shapes: Vec<Shape>,
}

fn handler<const NP: usize, T: Tag, S: Simulation<NP, T>>(
    mut simulation: S,
) -> (Sender<Request<T>>, Receiver<Reply>) {
    let (request_tx, mut request_rx) = channel::<Request<T>>(100);
    let (mut reply_tx, reply_rx) = channel::<Reply>(100);
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::spawn(move || {
        loop {
            let request = match request_rx.try_next() {
                Ok(Some(request)) => request,
                Ok(None) => {
                    return;
                }
                Err(_) => Request::Reinforce,
            };
            if let Some(reply) = simulation.handle_request(request) {
                if let Err(err) = reply_tx.try_send(reply) {
                    log::log!(log::Level::Error, "{err}");
                    return;
                }
            }
        }
    });
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async move {
        use futures::SinkExt;
        loop {
            let request = match request_rx.try_next() {
                Ok(Some(request)) => request,
                Ok(None) => {
                    return;
                }
                Err(_) => Request::Reinforce,
            };
            if let Some(reply) = simulation.handle_request(request) {
                if let Err(err) = reply_tx.send(reply).await {
                    log::log!(log::Level::Error, "{err}");
                    return;
                }
            }
            gloo_timers::future::TimeoutFuture::new(0).await;
        }
    });
    (request_tx, reply_rx)
}

impl<const NP: usize, T: Tag, S: Simulation<NP, T>, FS: Fn() -> S> SimulationGUI<NP, T, S, FS> {
    pub fn new(create_simulation: FS) -> Self {
        let simulation = create_simulation();
        let parameters = simulation.parameters();
        let (simulation_request_tx, simulation_reply_rx) = handler(simulation);
        SimulationGUI {
            parameters,
            simulation_request_tx,
            simulation_reply_rx,
            create_simulation,
            cached_shapes: Vec::new(),
        }
    }
}
impl<const NP: usize, T: Tag, S: Simulation<NP, T>, FS: Fn() -> S> eframe::App
    for SimulationGUI<NP, T, S, FS>
{
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut requests = vec![];
        if let Ok(Some(reply)) = self.simulation_reply_rx.try_next() {
            match reply {
                Reply::Shapes(shapes) => self.cached_shapes = shapes,
            }
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("reset").clicked() {
                    requests.push(Request::Reset);
                }
            });
            for p in self.parameters.iter_mut() {
                if ui
                    .add(
                        egui::Slider::new(&mut p.value, p.range.clone())
                            .logarithmic(p.logarithmic)
                            .text(p.tag.str()),
                    )
                    .changed()
                {
                    requests.push(Request::UpdateParameter(UpadeParameter {
                        tag: p.tag,
                        value: p.value,
                    }));
                }
            }

            Frame::canvas(ui.style()).show(ui, |ui| {
                let desired_size = ui.available_size();
                let ratio = desired_size.x / desired_size.y;
                let rx = ratio.max(1.0);
                let ry = ratio.recip().max(1.0);
                let (_id, rect) = ui.allocate_space(desired_size);

                let to_screen = RectTransform::from_to(
                    Rect::from_x_y_ranges(-200.0 * rx..=200.0 * rx, -200.0 * ry..=200.0 * ry),
                    rect,
                );
                requests.push(Request::Render(to_screen));

                ui.painter_at(rect).extend(self.cached_shapes.clone());
            });
        });
        for request in requests {
            if let Err(err) = self.simulation_request_tx.try_send(request) {
                log::log!(log::Level::Error, "{err}");
                let simulation = (self.create_simulation)();
                let (tx, rx) = handler(simulation);
                self.simulation_request_tx = tx;
                self.simulation_reply_rx = rx;
            }
        }
        ctx.request_repaint();
    }
}
