use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub mod camera;
pub mod gltf;
pub mod light;
pub mod model;

pub struct Scene {
    pub models: HashMap<String, model::Model>,
    pub camera: camera::Camera,
    pub lighting: light::Lighting,
}

impl Scene {
    pub fn new(aspect_ratio: f32) -> Self {
        Scene {
            models: HashMap::new(),
            camera: camera::Camera::new(aspect_ratio),
            lighting: light::Lighting::new(),
        }
    }

    pub fn load_model(&mut self, path: &Path, name: String) {
        let file = File::open(path).expect("engine: file doesn't exist");
        let reader = BufReader::new(file);
        let data: gltf::GLTF =
            serde_json::from_reader(reader).expect("engine: couldn't parse file");
        let model_dir = path.parent().unwrap();
        for mesh in &data.meshes {
            let mut primitives = Vec::new();
            for p in &mesh.primitives {
                let positions = load_positions(model_dir, &p, &data);
                let normals = load_normals(model_dir, &p, &data);
                let indices = load_indices(model_dir, &p, &data);
                let primitive = model::Primitive {
                    positions: positions,
                    normals: normals,
                    indices: indices,
                    material_index: p.material_index,
                };
                primitives.push(primitive);
            }
            let model = model::Model::new(primitives, data.materials.clone());
            self.models.insert(name.clone(), model);
        }
    }
}

fn load_positions(
    model_dir: &Path,
    primitive: &gltf::Primitive,
    data: &gltf::GLTF,
) -> Vec<model::Position> {
    let pos_acc_index = primitive.attributes.positions_accessor_index;
    let pos_acc = &data.accessors[pos_acc_index];
    let pos_buffer_view = &data.buffer_views[pos_acc.buffer_view_index];
    let pos_buffer = &data.buffers[pos_buffer_view.buffer_index];
    let buffer_path = model_dir.join(&pos_buffer.uri);
    let mut buffer_file = File::open(buffer_path).expect("engine: couldn't open mesh data file");
    _ = buffer_file.seek(SeekFrom::Start(pos_buffer_view.byte_offset));
    let mut byte_buffer = vec![0u8; pos_buffer_view.byte_length];
    buffer_file
        .read_exact(&mut byte_buffer)
        .expect("engine: couldn't read mesh data file");
    let mut positions: Vec<model::Position> = Vec::with_capacity(pos_acc.data_struct_count);
    for chunk in byte_buffer.chunks_exact(12) {
        let x = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
        positions.push(model::Position::new(x, y, z));
    }
    positions
}

fn load_normals(
    model_dir: &Path,
    primitive: &gltf::Primitive,
    data: &gltf::GLTF,
) -> Vec<model::Normal> {
    let pos_acc_index = primitive.attributes.normals_accessor_index;
    let pos_acc = &data.accessors[pos_acc_index];
    let pos_buffer_view = &data.buffer_views[pos_acc.buffer_view_index];
    let pos_buffer = &data.buffers[pos_buffer_view.buffer_index];
    let buffer_path = model_dir.join(&pos_buffer.uri);
    let mut buffer_file = File::open(buffer_path).expect("engine: couldn't open mesh data file");
    _ = buffer_file.seek(SeekFrom::Start(pos_buffer_view.byte_offset));
    let mut byte_buffer = vec![0u8; pos_buffer_view.byte_length];
    buffer_file
        .read_exact(&mut byte_buffer)
        .expect("engine: couldn't read mesh data file");
    let mut normals: Vec<model::Normal> = Vec::with_capacity(pos_acc.data_struct_count);
    for chunk in byte_buffer.chunks_exact(12) {
        let x = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
        normals.push(model::Normal::new(x, y, z));
    }
    normals
}

fn load_indices(model_dir: &Path, primitive: &gltf::Primitive, data: &gltf::GLTF) -> Vec<u16> {
    let acc_index = primitive.indices_accessor_index;
    let acc = &data.accessors[acc_index];
    let buffer_view = &data.buffer_views[acc.buffer_view_index];
    let buffer = &data.buffers[buffer_view.buffer_index];
    let buffer_path = model_dir.join(&buffer.uri);
    let mut buffer_file = File::open(buffer_path).expect("engine: couldn't open mesh data file");
    _ = buffer_file.seek(SeekFrom::Start(buffer_view.byte_offset));
    let mut byte_buffer = vec![0u8; buffer_view.byte_length];
    buffer_file
        .read_exact(&mut byte_buffer)
        .expect("engine: couldn't read mesh data file");
    let mut indices: Vec<u16> = Vec::with_capacity(acc.data_struct_count);
    for chunk in byte_buffer.chunks_exact(2) {
        let bytes: [u8; 2] = chunk.try_into().unwrap();
        indices.push(u16::from_le_bytes(bytes));
    }
    indices
}
