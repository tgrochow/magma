use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

#[derive(Clone)]
pub struct Texture {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub image_view: Option<Arc<ImageView>>,
    pub sampler: Option<Arc<Sampler>>,
}

impl Texture {
    pub fn new(gltf_path: &Path, uri: String) -> Self {
        let texture_path = gltf_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(uri);
        let rgba = image::open(&texture_path).unwrap().to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixels = rgba.into_raw();
        Texture {
            path: texture_path,
            width: width,
            height: height,
            pixels: pixels,
            image_view: None,
            sampler: None,
        }
    }

    pub fn load_texture(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        device: Arc<Device>,
    ) {
        let texture_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.pixels.clone(),
        )
        .unwrap();
        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [self.width, self.height, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                texture_buffer,
                image.clone(),
            ))
            .unwrap();
        self.image_view = Some(ImageView::new_default(image).unwrap());
        self.sampler = Some(
            Sampler::new(
                device.clone(),
                SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
            )
            .unwrap(),
        );
    }
}
