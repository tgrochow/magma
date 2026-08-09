use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::{ClearValue, Format};
use vulkano::image::SampleCount;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::render_pass::RenderPass;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo};
use vulkano::swapchain::Swapchain;

pub fn create_render_pass(
    device: Arc<Device>,
    swapchain: Arc<Swapchain>,
    sample_count: SampleCount,
) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: sample_count,
                load_op: Clear,
                store_op: DontCare,
            },
            resolve_color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: DontCare,
                store_op: Store,
            },
            depth_stencil: {
                format: Format::D16_UNORM,
                samples: sample_count,
                load_op: Clear,
                store_op: DontCare,
            },
        },
        pass: {
            color: [color],
            color_resolve: [resolve_color],
            depth_stencil: {depth_stencil},
        },
    )
    .unwrap()
}

pub fn create_framebuffers(
    memory_allocator: Arc<StandardMemoryAllocator>,
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    sample_count: SampleCount,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let depth_view = ImageView::new_default(
                Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::D16_UNORM,
                        extent: image.extent(),
                        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::TRANSIENT_ATTACHMENT,
                        samples: sample_count,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();
            let msaa_view = ImageView::new_default(
                Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: image.format(),
                        extent: image.extent(),
                        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                        samples: sample_count,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();
            let resolve_view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![msaa_view.clone(), resolve_view.clone(), depth_view.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

pub fn get_clear_values() -> Vec<Option<ClearValue>> {
    vec![Some([0.0, 0.0, 0.0, 1.0].into()), None, Some(1f32.into())]
}
