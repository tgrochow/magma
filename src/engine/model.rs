use glam::{Mat4, Vec3};
use std::f32::consts::TAU;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct Position {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct Normal {
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
}

pub struct Model {
    positions: Vec<Position>,
    normals: Vec<Normal>,
    indices: Vec<u16>,
    translation: Vec3,
    rotation_x: f32,
    rotation_y: f32,
    rotation_z: f32,
}

impl Model {
    pub fn new(positions: Vec<Position>, normals: Vec<Normal>, indeces: Vec<u16>) -> Self {
        Model {
            positions: positions,
            normals: normals,
            indices: indeces,
            translation: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation_x: 0.0,
            rotation_y: 0.0,
            rotation_z: 0.0,
        }
    }

    pub fn create_vertex_buffer(
        &self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) -> Subbuffer<[Position]> {
        Buffer::from_iter(
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
            self.positions.clone(),
        )
        .unwrap()
    }

    pub fn create_normals_buffer(
        &self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) -> Subbuffer<[Normal]> {
        Buffer::from_iter(
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
            self.normals.clone(),
        )
        .unwrap()
    }

    pub fn create_index_buffer(
        &self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) -> Subbuffer<[u16]> {
        Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.indices.clone(),
        )
        .unwrap()
    }

    pub fn get_model_matrix(&self) -> Mat4 {
        let mut model_matrix = Mat4::IDENTITY;
        if self.translation.length() > 0.0 {
            let m_t = Mat4::from_translation(self.translation);
            model_matrix *= m_t
        }
        if self.rotation_x != 0.0 {
            let m_rx = Mat4::from_rotation_x(self.rotation_x);
            model_matrix *= m_rx
        }
        if self.rotation_y != 0.0 {
            let m_ry = Mat4::from_rotation_y(self.rotation_y);
            model_matrix *= m_ry
        }
        if self.rotation_z != 0.0 {
            let m_rz = Mat4::from_rotation_z(self.rotation_z);
            model_matrix *= m_rz
        }
        return model_matrix;
    }

    pub fn rotate(&mut self, x: f32, y: f32, z: f32) {
        self.rotation_x = (self.rotation_x + x) % TAU;
        self.rotation_y = (self.rotation_y + y) % TAU;
        self.rotation_z = (self.rotation_z + z) % TAU;
    }

    pub fn translate(&mut self, vt: Vec3) {
        self.translation += vt;
    }
}

pub fn get_cube() -> Model {
    let positions = vec![
        Position {
            position: [-0.5, -0.5, 0.5],
        },
        Position {
            position: [0.5, -0.5, 0.5],
        },
        Position {
            position: [-0.5, 0.5, 0.5],
        },
        Position {
            position: [0.5, 0.5, 0.5],
        },
        Position {
            position: [-0.5, -0.5, -0.5],
        },
        Position {
            position: [0.5, -0.5, -0.5],
        },
        Position {
            position: [-0.5, 0.5, -0.5],
        },
        Position {
            position: [0.5, 0.5, -0.5],
        },
        Position {
            position: [-0.5, -0.5, 0.5],
        },
        Position {
            position: [-0.5, -0.5, -0.5],
        },
        Position {
            position: [-0.5, 0.5, 0.5],
        },
        Position {
            position: [-0.5, 0.5, -0.5],
        },
        Position {
            position: [0.5, -0.5, 0.5],
        },
        Position {
            position: [0.5, -0.5, -0.5],
        },
        Position {
            position: [0.5, 0.5, 0.5],
        },
        Position {
            position: [0.5, 0.5, -0.5],
        },
        Position {
            position: [-0.5, -0.5, -0.5],
        },
        Position {
            position: [0.5, -0.5, -0.5],
        },
        Position {
            position: [-0.5, -0.5, 0.5],
        },
        Position {
            position: [0.5, -0.5, 0.5],
        },
        Position {
            position: [-0.5, 0.5, -0.5],
        },
        Position {
            position: [0.5, 0.5, -0.5],
        },
        Position {
            position: [-0.5, 0.5, 0.5],
        },
        Position {
            position: [0.5, 0.5, 0.5],
        },
    ];
    let normals = vec![
        Normal {
            normal: [0.0, 0.0, 1.0],
        },
        Normal {
            normal: [0.0, 0.0, 1.0],
        },
        Normal {
            normal: [0.0, 0.0, 1.0],
        },
        Normal {
            normal: [0.0, 0.0, 1.0],
        },
        Normal {
            normal: [0.0, 0.0, -1.0],
        },
        Normal {
            normal: [0.0, 0.0, -1.0],
        },
        Normal {
            normal: [0.0, 0.0, -1.0],
        },
        Normal {
            normal: [0.0, 0.0, -1.0],
        },
        Normal {
            normal: [-1.0, 0.0, 0.0],
        },
        Normal {
            normal: [-1.0, 0.0, 0.0],
        },
        Normal {
            normal: [-1.0, 0.0, 0.0],
        },
        Normal {
            normal: [-1.0, 0.0, 0.0],
        },
        Normal {
            normal: [1.0, 0.0, 0.0],
        },
        Normal {
            normal: [1.0, 0.0, 0.0],
        },
        Normal {
            normal: [1.0, 0.0, 0.0],
        },
        Normal {
            normal: [1.0, 0.0, 0.0],
        },
        Normal {
            normal: [0.0, -1.0, 0.0],
        },
        Normal {
            normal: [0.0, -1.0, 0.0],
        },
        Normal {
            normal: [0.0, -1.0, 0.0],
        },
        Normal {
            normal: [0.0, -1.0, 0.0],
        },
        Normal {
            normal: [0.0, 1.0, 0.0],
        },
        Normal {
            normal: [0.0, 1.0, 0.0],
        },
        Normal {
            normal: [0.0, 1.0, 0.0],
        },
        Normal {
            normal: [0.0, 1.0, 0.0],
        },
    ];
    let indices = vec![
        0, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17,
        18, 17, 18, 19, 20, 21, 22, 21, 22, 23,
    ];
    Model::new(positions, normals, indices)
}
