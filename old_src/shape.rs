use std::sync::Arc;
use vulkano::buffer::Buffer;
use vulkano::buffer::BufferContents;
use vulkano::buffer::BufferCreateInfo;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::Subbuffer;
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::memory::allocator::FreeListAllocator;
use vulkano::memory::allocator::GenericMemoryAllocator;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::shader::EntryPoint;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct ShapeVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

pub fn get_test_triangle(
    memory_allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,
) -> Subbuffer<[ShapeVertex]> {
    let vertex1 = ShapeVertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = ShapeVertex {
        position: [0.0, 0.5],
    };
    let vertex3 = ShapeVertex {
        position: [0.5, -0.25],
    };
    return Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    )
    .unwrap();
}

pub fn get_vertex_input_state(entry_point: &EntryPoint) -> VertexInputState {
    return ShapeVertex::per_vertex().definition(&entry_point).unwrap();
}
