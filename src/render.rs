use crate::mesh::MeshVertex;
use crate::{device, mesh, render, shader};
use std::sync::Arc;
use vulkano::image::ImageUsage;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::SwapchainCreateInfo;
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

pub struct RenderState {
    pub window: Arc<Window>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub viewport: Viewport,
    pub vs: Arc<ShaderModule>,
    pub fs: Arc<ShaderModule>,
    pub render_pass: Arc<RenderPass>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub vertex_buffer: Subbuffer<[MeshVertex]>,
    pub command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub swapchain: Arc<Swapchain>,
    pub fences: Vec<
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
    pub previous_fence_i: u32,
    pub window_resized: bool,
    pub recreate_swapchain: bool,
}

impl RenderState {
    pub fn window_resized(&mut self) {
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
                let new_framebuffers = render::get_framebuffers(&new_images, &self.render_pass);
                self.viewport.extent = new_dimensions.into();
                let new_pipeline = render::get_pipeline(
                    self.device.clone(),
                    self.vs.clone(),
                    self.fs.clone(),
                    self.render_pass.clone(),
                    self.viewport.clone(),
                );
                self.command_buffers = render::get_command_buffers(
                    &self.command_buffer_allocator,
                    &self.queue,
                    &new_pipeline,
                    &new_framebuffers,
                    &self.vertex_buffer,
                );
            }
        }
    }
}

pub fn init(instance: &Arc<Instance>, window: Arc<Window>) -> RenderState {
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
    return RenderState {
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
    };
}

pub fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
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

pub fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
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

pub fn get_pipeline(
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

pub fn get_command_buffers(
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
