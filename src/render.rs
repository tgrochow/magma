use crate::mesh::MeshVertex;
use crate::{device, mesh, shader};
use std::sync::Arc;
use vulkano::image::ImageUsage;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::{self, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync;
use vulkano::{Validated, VulkanError};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo, allocator::StandardCommandBufferAllocator,
    },
    device::{Device, Queue},
    image::{Image, view::ImageView},
    instance::Instance,
    pipeline::{
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{PresentFuture, Surface, Swapchain, SwapchainAcquireFuture},
    sync::{
        GpuFuture,
        future::{FenceSignalFuture, JoinFuture},
    },
};
use winit::window::Window;

pub struct Engine {
    pub window: Arc<Window>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    viewport: Viewport,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MeshVertex]>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    swapchain: Arc<Swapchain>,
    fences: Vec<
        Option<
            Arc<
                FenceSignalFuture<
                    PresentFuture<
                        CommandBufferExecFuture<
                            JoinFuture<Box<dyn GpuFuture + 'static>, SwapchainAcquireFuture>,
                        >,
                    >,
                >,
            >,
        >,
    >,
    previous_fence_index: u32,
    pub window_resized: bool,
    recreate_swapchain: bool,
}

impl Engine {
    pub fn handle_window_resize(&mut self) {
        if self.window_resized || self.recreate_swapchain {
            self.recreate_swapchain = false;
            let new_dimensions = self.window.inner_size();
            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    // Here, `image_extend` will correspond to the window dimensions.
                    image_extent: new_dimensions.into(),
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain: {e}");
            self.swapchain = new_swapchain;
            if self.window_resized {
                self.window_resized = false;
                let new_framebuffers = get_framebuffers(&new_images, &self.render_pass);
                self.viewport.extent = new_dimensions.into();
                let new_pipeline = get_pipeline(
                    self.device.clone(),
                    self.vs.clone(),
                    self.fs.clone(),
                    self.render_pass.clone(),
                    self.viewport.clone(),
                );
                self.command_buffers = get_command_buffers(
                    &self.command_buffer_allocator,
                    &self.queue,
                    &new_pipeline,
                    &new_framebuffers,
                    &self.vertex_buffer,
                );
            }
        }
    }

    pub fn draw(&mut self) {
        self.handle_window_resize();
        let (image_index, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };
        if suboptimal {
            self.recreate_swapchain = true;
        }
        // Wait for the fence related to this image to finish. Normally this would be the
        // oldest fence that most likely has already finished.
        if let Some(image_fence) = &self.fences[image_index as usize] {
            image_fence.wait(None).unwrap();
        }
        let previous_future = match self.fences[self.previous_fence_index as usize].clone() {
            // Create a `NowFuture`.
            None => {
                let mut now = sync::now(self.device.clone());
                now.cleanup_finished();
                now.boxed()
            }
            // Use the existing `FenceSignalFuture`.
            Some(fence) => fence.boxed(),
        };
        let future = previous_future
            .join(acquire_future)
            .then_execute(
                self.queue.clone(),
                self.command_buffers[image_index as usize].clone(),
            )
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();
        self.fences[image_index as usize] = match future.map_err(Validated::unwrap) {
            Ok(value) => Some(Arc::new(value)),
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                None
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                None
            }
        };
        self.previous_fence_index = image_index;
    }
}

pub fn init(instance: &Arc<Instance>, window: Arc<Window>) -> Engine {
    let surface = Surface::from_window(instance.clone(), window.clone())
        .expect("surface could not be created");
    let (physical_device, device, queue) = device::init_device(instance, &surface);
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
    let render_pass = get_render_pass(device.clone(), &sc);
    let framebuffers = get_framebuffers(&images, &render_pass);
    let vs = shader::vs::load(device.clone()).expect("failed to create shader module");
    let fs = shader::fs::load(device.clone()).expect("failed to create shader module");
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };
    let pipeline = get_pipeline(
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
    let command_buffers = get_command_buffers(
        &command_buffer_allocator,
        &queue,
        &pipeline,
        &framebuffers,
        &vertex_buffer,
    );
    let frames_in_flight = images.len();
    return Engine {
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
        previous_fence_index: 0,
        window_resized: false,
        recreate_swapchain: false,
    };
}

fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                // Set the format the same as the swapchain.
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap()
}

fn get_framebuffers(images: &[Arc<Image>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();
    let vertex_input_state = mesh::get_vertex_input_state(&vs);
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn get_command_buffers(
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: &Subbuffer<[mesh::MeshVertex]>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            unsafe {
                let mut builder = AutoCommandBufferBuilder::primary(
                    command_buffer_allocator.clone(),
                    queue.queue_family_index(),
                    // Don't forget to write the correct buffer usage.
                    CommandBufferUsage::MultipleSubmit,
                )
                .unwrap();
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(SubpassEndInfo::default())
                    .unwrap();
                builder.build().unwrap()
            }
        })
        .collect()
}
