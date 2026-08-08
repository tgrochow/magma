// pub fn get_render_pass() {
// let render_pass = vulkano::single_pass_renderpass!(
//     device.clone(),
//     attachments: {
//         color: {
//             format: swapchain.image_format(),
//             samples: SampleCount::Sample4,
//             load_op: Clear,
//             store_op: DontCare,
//         },
//         resolve_color: {
//             format: swapchain.image_format(),
//             samples: 1,
//             load_op: DontCare,
//             store_op: Store,
//         },
//         depth_stencil: {
//             format: Format::D16_UNORM,
//             samples: SampleCount::Sample4,
//             load_op: Clear,
//             store_op: DontCare,
//         },
//     },
//     pass: {
//         color: [color],
//         color_resolve: [resolve_color],
//         depth_stencil: {depth_stencil},
//     },
// )
// .unwrap();
// }
