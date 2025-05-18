use std::sync::Arc;
use vulkano::{device::Device, format::Format, image::view::ImageView};

fn render(device: &Arc<Device>) {
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: Format::R8G8B8A8_UNORM,
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
    .unwrap();
    // let view = ImageView::new_default(image.clone()).unwrap();
    // let framebuffer = Framebuffer::new(
    //     render_pass.clone(),
    //     FramebufferCreateInfo {
    //         attachments: vec![view],
    //         ..Default::default()
    //     },
    // )
    // .unwrap();
}
