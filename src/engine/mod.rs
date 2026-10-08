use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::PrimaryCommandBufferAbstract;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount};
use vulkano::instance::Instance;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::EntryPoint;
use vulkano::swapchain::{
    Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::scene::{self, texture};
use crate::shader;

mod config;
mod device;
mod msaa;
mod statistic;

pub struct Engine {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    window_size: PhysicalSize<u32>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    vertex_shader: EntryPoint,
    fragment_shader: EntryPoint,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swapchain: bool,
    config: config::Config,
    statistik: statistic::Statistic,
}

impl Engine {
    pub fn new(instance: &Arc<Instance>, window: Arc<Window>, sample_count: SampleCount) -> Self {
        let surface = Surface::from_window(instance.clone(), window.clone())
            .expect("engine: surface could not be created");
        let (physical_device, device, queue) = device::init_device(instance, &surface);
        let config = config::Config::new(physical_device, sample_count);
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let window_size = window.inner_size();
        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let (image_format, _) = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];
            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let render_pass = create_render_pass(device.clone(), swapchain.clone(), sample_count);
        let framebuffers = create_framebuffers(
            memory_allocator.clone(),
            &images,
            render_pass.clone(),
            sample_count,
        );
        let vertex_shader = shader::pbr_vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fragment_shader = shader::pbr_fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let pipeline = create_pipeline(
            device.clone(),
            render_pass.clone(),
            vertex_shader.clone(),
            fragment_shader.clone(),
            window_size,
            sample_count,
        );
        let previous_frame_end = Some(sync::now(device.clone()).boxed());
        Engine {
            device: device,
            queue: queue,
            memory_allocator: memory_allocator,
            descriptor_set_allocator: descriptor_set_allocator,
            command_buffer_allocator: command_buffer_allocator,
            uniform_buffer_allocator: uniform_buffer_allocator,
            window_size: window_size,
            swapchain: swapchain,
            render_pass: render_pass,
            vertex_shader: vertex_shader,
            fragment_shader: fragment_shader,
            framebuffers: framebuffers,
            pipeline: pipeline,
            previous_frame_end: previous_frame_end,
            recreate_swapchain: false,
            config: config,
            statistik: statistic::Statistic::new(),
        }
    }

    pub fn prepare_scene(&mut self, scene: &mut scene::Scene) {
        let mut upload_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        for texture in &mut scene.textures {
            texture.load_texture(
                &mut upload_builder,
                self.memory_allocator.clone(),
                self.device.clone(),
            );
        }
        let upload_command_buffer = upload_builder.build().unwrap();
        upload_command_buffer
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    pub fn draw(&mut self, scene: &mut scene::Scene, window_size: PhysicalSize<u32>) {
        if window_size.width == 0 || window_size.height == 0 {
            return;
        }
        if self.recreate_swapchain {
            self.update_window_size(window_size);
        }
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("engine: failed to acquire next image: {e}"),
            };
        if suboptimal {
            self.recreate_swapchain = true;
        }
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        // important: call cleanup_finished before begin_render_pass
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: get_clear_values(self.config.sample_count),
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                Default::default(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap();
        for (_key, model) in &scene.models {
            self.draw_model(
                &mut builder,
                &model,
                &scene.camera,
                &scene.lighting,
                &scene.textures,
            );
        }
        builder.end_render_pass(Default::default()).unwrap();
        let command_buffer = builder.build().unwrap();
        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();
        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
                self.statistik.tick();
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("engine: failed to flush future: {e}");
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }

    fn draw_model(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        model: &scene::model::Model,
        camera: &scene::camera::Camera,
        lighting: &scene::light::Lighting,
        textures: &Vec<texture::Texture>,
    ) {
        for primitive in &model.primitives {
            self.draw_primitive(
                builder,
                model,
                primitive,
                camera,
                &model.materials[primitive.material_index],
                lighting,
                textures,
            );
        }
    }

    fn draw_primitive(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        model: &scene::model::Model,
        primitive: &scene::model::Primitive,
        camera: &scene::camera::Camera,
        material: &scene::material::Material,
        lighting: &scene::light::Lighting,
        textures: &Vec<texture::Texture>,
    ) {
        let pos_buffer = primitive.create_vertex_buffer(&self.memory_allocator);
        let normals_buffer = primitive.create_normals_buffer(&self.memory_allocator);
        let texture_coords_buffer =
            primitive.create_texture_coordinates_buffer(&self.memory_allocator);
        let index_buffer = primitive.create_index_buffer(&self.memory_allocator);
        let index_buffer_length = index_buffer.len() as u32;
        let vs_uniform_buffer = {
            let uniform_data = shader::pbr_vs::Data {
                world: model.get_model_matrix().to_cols_array_2d(),
                view: camera.get_view_matrix().to_cols_array_2d(),
                proj: camera.get_projection_matrix().to_cols_array_2d(),
            };
            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;
            buffer
        };
        let fs_uniform_buffer = {
            let pos = camera.get_position();
            let uniform_data = shader::pbr_fs::CameraData {
                pos: pos.to_array(),
            };
            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;
            buffer
        };
        let point_lights_buffer = lighting.create_point_light_buffer(self.memory_allocator.clone());
        let dir_lights_buffer = lighting.create_dir_light_buffer(self.memory_allocator.clone());
        let texture_index = material.color_texture_index;
        let texture_view = textures[texture_index].image_view.as_ref().unwrap().clone();
        let sampler = textures[texture_index].sampler.as_ref().unwrap().clone();
        let layout = &self.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, vs_uniform_buffer),
                WriteDescriptorSet::buffer(1, fs_uniform_buffer),
                WriteDescriptorSet::buffer(2, point_lights_buffer),
                WriteDescriptorSet::buffer(3, dir_lights_buffer),
                WriteDescriptorSet::image_view_sampler(4, texture_view, sampler),
            ],
            [],
        )
        .unwrap();
        let push_constants = shader::pbr_fs::PushConstantData {
            useColorTexture: material.use_color_texture,
            materialColor: material.color,
            materialMettalic: material.mettalic,
            materialRoughness: material.roughness,
            pointLightCount: lighting.point_ligths.len() as u32,
            dirLightCount: lighting.dir_lights.len() as u32,
        };
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .bind_vertex_buffers(0, (pos_buffer, normals_buffer, texture_coords_buffer))
            .unwrap()
            .bind_index_buffer(index_buffer)
            .unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .unwrap();
        unsafe { builder.draw_indexed(index_buffer_length, 1, 0, 0, 0) }.unwrap();
    }

    fn update_window_size(&mut self, window_size: PhysicalSize<u32>) {
        self.window_size = window_size;
        self.recreate_swapchain = false;
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: window_size.into(),
                ..self.swapchain.create_info()
            })
            .expect("engine: failed to recreate swapchain");
        self.swapchain = new_swapchain;
        let new_framebuffers = create_framebuffers(
            self.memory_allocator.clone(),
            &new_images,
            self.render_pass.clone(),
            self.config.sample_count,
        );
        let new_pipeline = create_pipeline(
            self.device.clone(),
            self.render_pass.clone(),
            self.vertex_shader.clone(),
            self.fragment_shader.clone(),
            window_size,
            self.config.sample_count,
        );
        self.framebuffers = new_framebuffers;
        self.pipeline = new_pipeline;
    }

    pub fn recreate_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }

    pub fn get_fps(&self) -> i32 {
        return self.statistik.calc_fps();
    }
}

fn create_framebuffers(
    memory_allocator: Arc<StandardMemoryAllocator>,
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    sample_count: SampleCount,
) -> Vec<Arc<Framebuffer>> {
    if sample_count == SampleCount::Sample1 {
        create_framebuffers_without_msaa(memory_allocator.clone(), &images, render_pass.clone())
    } else {
        msaa::create_framebuffers(
            memory_allocator.clone(),
            &images,
            render_pass.clone(),
            sample_count,
        )
    }
}

fn create_framebuffers_without_msaa(
    memory_allocator: Arc<StandardMemoryAllocator>,
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let depth_view = ImageView::new_default(
                Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::D32_SFLOAT,
                        extent: image.extent(),
                        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::TRANSIENT_ATTACHMENT,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view.clone(), depth_view.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn create_render_pass(
    device: Arc<Device>,
    swapchain: Arc<Swapchain>,
    sample_count: SampleCount,
) -> Arc<RenderPass> {
    if sample_count == SampleCount::Sample1 {
        create_render_pass_without_msaa(device.clone(), swapchain.clone())
    } else {
        msaa::create_render_pass(device.clone(), swapchain.clone(), sample_count)
    }
}

fn create_render_pass_without_msaa(
    device: Arc<Device>,
    swapchain: Arc<Swapchain>,
) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth_stencil: {
                format: Format::D32_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {depth_stencil},
        },
    )
    .unwrap()
}

fn create_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    vs: EntryPoint,
    fs: EntryPoint,
    window_size: PhysicalSize<u32>,
    sample_count: SampleCount,
) -> Arc<GraphicsPipeline> {
    let vertex_input_state = [
        scene::model::Position::per_vertex(),
        scene::model::Normal::per_vertex(),
        scene::model::TextureCoords::per_vertex(),
    ]
    .definition(&vs)
    .unwrap();
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
                viewports: [Viewport {
                    offset: [0.0, 0.0],
                    extent: window_size.into(),
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState {
                rasterization_samples: sample_count,
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some((subpass).into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn get_clear_values(sample_count: SampleCount) -> Vec<Option<ClearValue>> {
    if sample_count == SampleCount::Sample1 {
        get_clear_values_without_msaa()
    } else {
        msaa::get_clear_values()
    }
}

fn get_clear_values_without_msaa() -> Vec<Option<ClearValue>> {
    vec![Some([0.8, 0.8, 0.8, 1.0].into()), Some(1f32.into())]
}
