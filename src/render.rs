use std::sync::Arc;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        CommandBufferExecFuture, PrimaryAutoCommandBuffer,
        allocator::StandardCommandBufferAllocator,
    },
    device::{Device, Queue},
    image::{Image, view::ImageView},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    shader::ShaderModule,
    swapchain::{PresentFuture, Swapchain, SwapchainAcquireFuture},
    sync::{
        GpuFuture,
        future::{FenceSignalFuture, JoinFuture},
    },
};
use winit::window::Window;

use crate::mesh::MeshVertex;

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
