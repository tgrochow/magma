use crate::engine::model::{Model, Normal, Position};

use glam::Vec3;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Deserialize)]
struct GLTF {
    meshes: Vec<GLTFMesh>,
    accessors: Vec<GLTFAccessor>,
    #[serde(rename = "bufferViews")]
    buffer_views: Vec<GLTFBufferView>,
    buffers: Vec<GLTFBuffer>,
}

#[derive(Deserialize)]
struct GLTFMesh {
    name: String,
    primitives: Vec<GLTFPrimitives>,
}

#[derive(Deserialize)]
struct GLTFPrimitives {
    attributes: GLTFAttributes,
    #[serde(rename = "indices")]
    indices_accessor_index: usize,
}

#[derive(Deserialize)]
struct GLTFAttributes {
    #[serde(rename = "POSITION")]
    positions_accessor_index: usize,
    #[serde(rename = "NORMAL")]
    normals_accessor_index: usize,
    #[serde(rename = "TEXCOORD_0")]
    texcoords_accessor_index: usize,
}

#[derive(Deserialize)]
struct GLTFAccessor {
    #[serde(rename = "bufferView")]
    buffer_view_index: usize,
    #[serde(rename = "componentType")]
    data_type: u16,
    #[serde(rename = "type")]
    data_struct_type: String,
    #[serde(rename = "count")]
    data_struct_count: usize,
}

#[derive(Deserialize)]
struct GLTFBufferView {
    #[serde(rename = "buffer")]
    buffer_index: usize,
    #[serde(rename = "byteLength")]
    byte_length: usize,
    #[serde(rename = "byteOffset")]
    byte_offset: u64,
}

#[derive(Deserialize)]
struct GLTFBuffer {
    #[serde(rename = "byteLength")]
    byte_length: u64,
    uri: String,
}

pub struct Scene {
    pub models: HashMap<String, Model>,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            models: HashMap::new(),
        }
    }

    pub fn load_model(&mut self, path: &Path) {
        let file = File::open(path).expect("engine: file doesn't exist");
        let reader = BufReader::new(file);
        let data: GLTF = serde_json::from_reader(reader).expect("engine: couldn't parse file");
        let model_dir = path.parent().unwrap();
        for mesh in &data.meshes {
            println!("scene: load model {}", mesh.name);
            let positions = load_positions(model_dir, &mesh, &data);
            let normals = load_normals(model_dir, &mesh, &data);
            let indices = load_indices(model_dir, &mesh, &data);
            let mut model = Model::new(positions, normals, indices);
            model.translate(Vec3 {
                x: 0.0,
                y: 0.0,
                z: -3.0,
            });
            self.models.insert("adssad".to_string(), model);
        }
    }
}

fn load_positions(model_dir: &Path, mesh: &GLTFMesh, data: &GLTF) -> Vec<Position> {
    let pos_acc_index = mesh.primitives[0].attributes.positions_accessor_index;
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
    let mut positions: Vec<Position> = Vec::with_capacity(pos_acc.data_struct_count);
    for chunk in byte_buffer.chunks_exact(12) {
        let x = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
        positions.push(Position::new(x, y, z));
    }
    positions
}

fn load_normals(model_dir: &Path, mesh: &GLTFMesh, data: &GLTF) -> Vec<Normal> {
    let pos_acc_index = mesh.primitives[0].attributes.normals_accessor_index;
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
    let mut normals: Vec<Normal> = Vec::with_capacity(pos_acc.data_struct_count);
    for chunk in byte_buffer.chunks_exact(12) {
        let x = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
        normals.push(Normal::new(x, y, z));
    }
    normals
}

fn load_indices(model_dir: &Path, mesh: &GLTFMesh, data: &GLTF) -> Vec<u16> {
    let acc_index = mesh.primitives[0].indices_accessor_index;
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
