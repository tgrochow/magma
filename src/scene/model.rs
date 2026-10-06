use crate::scene::material;
use crate::scene::texture;
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
    pub position: [f32; 3],
}

impl Position {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Position {
            position: [x, y, z],
        }
    }
}

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct Normal {
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
}

impl Normal {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Normal { normal: [x, y, z] }
    }
}

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct TextureCoords {
    #[format(R32G32_SFLOAT)]
    pub texture_coords: [f32; 2],
}

impl TextureCoords {
    pub fn new(u: f32, w: f32) -> Self {
        TextureCoords {
            texture_coords: [u, w],
        }
    }
}

pub struct Primitive {
    pub positions: Vec<Position>,
    pub normals: Vec<Normal>,
    pub indices: Vec<u16>,
    pub tex_coords: Vec<TextureCoords>,
    pub material_index: usize,
}

impl Primitive {
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

    pub fn create_texture_coordinates_buffer(
        &self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) -> Subbuffer<[TextureCoords]> {
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
            self.tex_coords.clone(),
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
}

pub struct Model {
    pub primitives: Vec<Primitive>,
    pub materials: Vec<material::Material>,
    pub textures: Vec<texture::Texture>,
    translation: Vec3,
    rotation_x: f32,
    rotation_y: f32,
    rotation_z: f32,
}

impl Model {
    pub fn new(
        primitives: Vec<Primitive>,
        materials: Vec<material::Material>,
        textures: Vec<texture::Texture>,
    ) -> Self {
        Model {
            primitives: primitives,
            materials: materials,
            textures: textures,
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
