use glam::Vec3;
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

struct State {
    window: Arc<Window>,
    engine: engine::Engine,
    scene: scene::Scene,
    fps_updated: Instant,
}

struct App {
    instance: Arc<Instance>,
    state: Option<State>,
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
            state: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("Magma v0.1.0");
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        let window_size = window.inner_size();
        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let engine = engine::Engine::new(&self.instance, window.clone(), SampleCount::Sample4);
        let mut scene = scene::Scene::new(aspect_ratio, Vec3::new(0.0, 3.0, 12.0));
        scene.lighting.point_ligths.push(scene::light::PointLight {
            pos: [2.0, 4.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
        });
        scene.lighting.dir_lights.push(scene::light::DirLight {
            dir: [0.0, 0.0, -1.0, 1.0],
            color: [1.0, 1.0, 1.0, 2.0],
        });
        scene.load_model(Path::new("./models/fox/fox.gltf"), "fox01".to_string());
        self.state = Some(State {
            window,
            engine,
            scene,
            fps_updated: Instant::now(),
        });
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
                let state = self.state.as_mut().unwrap();
                let window_size = state.window.inner_size();
                let aspect_ratio = window_size.width as f32 / window_size.height as f32;
                state.engine.recreate_swapchain();
                state.scene.camera.update_projection(aspect_ratio);
            }
            WindowEvent::RedrawRequested => {
                let state = self.state.as_mut().unwrap();
                state.engine.draw(&state.scene, state.window.inner_size());
                state
                    .scene
                    .models
                    .get_mut("fox01")
                    .unwrap()
                    .rotate(0.0, 0.015, 0.0);
                if state.fps_updated.elapsed().as_secs() >= 3 {
                    let fps = state.engine.get_fps();
                    if fps > 0 {
                        let title = format!("Magma v0.1.0 - FPS: {}", fps);
                        state.window.set_title(&title);
                        state.fps_updated = Instant::now();
                    }
                }
                state.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let state = self.state.as_mut().unwrap();
                if event.state == ElementState::Pressed {
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        Key::Named(NamedKey::ArrowUp) => state.scene.camera.move_forward(),
                        Key::Named(NamedKey::ArrowDown) => state.scene.camera.move_backwards(),
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
