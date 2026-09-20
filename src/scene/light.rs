use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

pub struct Lighting {
    pub point_ligths: Vec<PointLight>,
    pub dir_lights: Vec<DirLight>,
}

impl Lighting {
    pub fn new() -> Self {
        Lighting {
            point_ligths: Vec::new(),
            dir_lights: Vec::new(),
        }
    }

    pub fn create_light_buffer(
        &self,
        memory_allocator: Arc<StandardMemoryAllocator>,
    ) -> vulkano::buffer::Subbuffer<[PointLight]> {
        Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.point_ligths.iter().copied(),
        )
        .expect("failed to create light buffer")
    }
}

#[repr(C)]
#[derive(BufferContents, Clone, Copy)]
pub struct PointLight {
    pub pos: [f32; 4],
    pub color: [f32; 4],
}

pub struct DirLight {
    pub dir: [f32; 3],
    pub color: [f32; 3],
}
