use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use vulkano::VulkanLibrary;
use vulkano::image::SampleCount;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

mod engine;
mod scene;
mod shader;

struct App {
    instance: Arc<Instance>,
    window: Option<Arc<Window>>,
    engine: Option<engine::Engine>,
    scene: Option<scene::Scene>,
}

struct Statistic {
    frame_rates: [i32; 10],
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(&event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create Vulkan instance");
        App {
            instance: instance,
            window: None,
            engine: None,
            scene: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("Magma v0.1.0");
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        let window_size = window.inner_size();
        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        self.window = Some(window);
        self.engine = Some(engine::Engine::new(
            &self.instance,
            self.window.as_ref().unwrap().clone(),
            SampleCount::Sample1,
        ));
        let mut scene = scene::Scene::new(aspect_ratio);
        scene.load_model(Path::new("./models/well.gltf"), "well".to_string());
        self.scene = Some(scene);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                println!("reszied");
                let window_size = self.window.as_ref().unwrap().inner_size();
                let aspect_ratio = window_size.width as f32 / window_size.height as f32;
                self.engine.as_mut().unwrap().recreate_swapchain();
                self.scene
                    .as_mut()
                    .unwrap()
                    .camera
                    .update_projection(aspect_ratio);
            }
            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();
                self.engine.as_mut().unwrap().draw(
                    &self.scene.as_ref().unwrap(),
                    self.window.as_mut().unwrap().inner_size(),
                );
                let frame_rate = 1.0 / frame_start.elapsed().as_secs_f32();
                let title = format!("FPS: {}", frame_rate.round() as u32);
                self.window.as_mut().unwrap().set_title(&title);
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        Key::Named(NamedKey::ArrowUp) => {
                            self.scene.as_mut().unwrap().camera.move_forward()
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            self.scene.as_mut().unwrap().camera.move_backwards()
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app)
}
