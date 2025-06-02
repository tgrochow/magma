use std::sync::Arc;
use vulkano::VulkanLibrary;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;

use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::swapchain;
use vulkano::swapchain::Surface;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::SwapchainCreateInfo;
use vulkano::swapchain::SwapchainPresentInfo;
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano::{Validated, VulkanError};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ControlFlow;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

mod device;
mod mesh;
mod render;
mod shader;

struct App {
    instance: Arc<Instance>,
    render_state: Option<render::RenderState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone())
            .expect("surface could not be created");
        let (physical_device, device, queue) = device::init_device(&self.instance, &surface);
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");
        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        let (sc, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1, // How many buffers to use in the swapchain
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT, // What the images are going to be used for
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();
        let render_pass = render::get_render_pass(device.clone(), &sc);
        let framebuffers = render::get_framebuffers(&images, &render_pass);
        let vs = shader::vs::load(device.clone()).expect("failed to create shader module");
        let fs = shader::fs::load(device.clone()).expect("failed to create shader module");
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };
        let pipeline = render::get_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        );
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let vertex_buffer = mesh::get_test_triangle(&memory_allocator);
        let command_buffers = render::get_command_buffers(
            &command_buffer_allocator,
            &queue,
            &pipeline,
            &framebuffers,
            &vertex_buffer,
        );
        let frames_in_flight = images.len();
        self.render_state = Some(render::RenderState {
            window: window,
            device: device,
            queue: queue,
            viewport: viewport,
            vs: vs,
            fs: fs,
            swapchain: sc,
            render_pass: render_pass,
            command_buffer_allocator: command_buffer_allocator,
            vertex_buffer: vertex_buffer,
            command_buffers: command_buffers,
            fences: vec![None; frames_in_flight],
            previous_fence_i: 0,
            window_resized: false,
            recreate_swapchain: false,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                let render_state = self.render_state.as_mut().unwrap();
                render_state.window_resized = true;
            }
            WindowEvent::RedrawRequested => {
                let render_state = self.render_state.as_mut().unwrap();
                if render_state.window_resized || render_state.recreate_swapchain {
                    render_state.recreate_swapchain = false;
                    let new_dimensions = render_state.window.inner_size();
                    let (new_swapchain, new_images) = render_state
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            // Here, `image_extend` will correspond to the window dimensions.
                            image_extent: new_dimensions.into(),
                            ..render_state.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain: {e}");
                    render_state.swapchain = new_swapchain;
                    if render_state.window_resized {
                        render_state.window_resized = false;
                        let new_framebuffers =
                            render::get_framebuffers(&new_images, &render_state.render_pass);
                        render_state.viewport.extent = new_dimensions.into();
                        let new_pipeline = render::get_pipeline(
                            render_state.device.clone(),
                            render_state.vs.clone(),
                            render_state.fs.clone(),
                            render_state.render_pass.clone(),
                            render_state.viewport.clone(),
                        );
                        render_state.command_buffers = render::get_command_buffers(
                            &render_state.command_buffer_allocator,
                            &render_state.queue,
                            &new_pipeline,
                            &new_framebuffers,
                            &render_state.vertex_buffer,
                        );
                    }
                }
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                // Draw.

                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw in
                // applications which do not always need to. Applications that redraw continuously
                // can render here instead.
                let (image_i, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(render_state.swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            render_state.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };
                if suboptimal {
                    render_state.recreate_swapchain = true;
                }
                // Wait for the fence related to this image to finish. Normally this would be the
                // oldest fence that most likely has already finished.
                if let Some(image_fence) = &render_state.fences[image_i as usize] {
                    image_fence.wait(None).unwrap();
                }
                let previous_future =
                    match render_state.fences[render_state.previous_fence_i as usize].clone() {
                        // Create a `NowFuture`.
                        None => {
                            let mut now = sync::now(render_state.device.clone());
                            now.cleanup_finished();

                            now.boxed()
                        }
                        // Use the existing `FenceSignalFuture`.
                        Some(fence) => fence.boxed(),
                    };

                let future = previous_future
                    .join(acquire_future)
                    .then_execute(
                        render_state.queue.clone(),
                        render_state.command_buffers[image_i as usize].clone(),
                    )
                    .unwrap()
                    .then_swapchain_present(
                        render_state.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            render_state.swapchain.clone(),
                            image_i,
                        ),
                    )
                    .then_signal_fence_and_flush();
                render_state.fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                    Ok(value) => Some(Arc::new(value)),
                    Err(VulkanError::OutOfDate) => {
                        render_state.recreate_swapchain = true;
                        None
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        None
                    }
                };
                render_state.previous_fence_i = image_i;
                render_state.window.request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    let (instance, event_loop) = init_vulkan();
    let mut app = App {
        instance: instance,
        render_state: None,
    };
    event_loop.run_app(&mut app).expect("fuck");
}

fn init_vulkan() -> (Arc<Instance>, EventLoop<()>) {
    let event_loop = EventLoop::new().unwrap();
    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);
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
    .expect("failed to create instance");
    (instance, event_loop)
}
