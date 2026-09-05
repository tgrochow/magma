use serde::Deserialize;

#[derive(Deserialize)]
pub struct GLTF {
    pub materials: Vec<Material>,
    pub meshes: Vec<Mesh>,
    pub accessors: Vec<Accessor>,
    #[serde(rename = "bufferViews")]
    pub buffer_views: Vec<BufferView>,
    pub buffers: Vec<Buffer>,
}

#[derive(Clone, Deserialize)]
pub struct Material {
    pub name: String,
    #[serde(rename = "pbrMetallicRoughness")]
    pub pbr: PBR,
}

#[derive(Clone, Deserialize)]
pub struct PBR {
    #[serde(rename = "baseColorFactor")]
    pub color: [f32; 4],
    #[serde(rename = "metallicFactor")]
    pub mettalic: f32,
    #[serde(rename = "roughnessFactor")]
    pub roughness: f32,
}

#[derive(Deserialize)]
pub struct Mesh {
    pub name: String,
    pub primitives: Vec<Primitive>,
}

#[derive(Deserialize)]
pub struct Primitive {
    pub attributes: Attributes,
    #[serde(rename = "indices")]
    pub indices_accessor_index: usize,
    #[serde(rename = "material")]
    pub material_index: usize,
}

#[derive(Deserialize)]
pub struct Attributes {
    #[serde(rename = "POSITION")]
    pub positions_accessor_index: usize,
    #[serde(rename = "NORMAL")]
    pub normals_accessor_index: usize,
    #[serde(rename = "TEXCOORD_0")]
    pub texcoords_accessor_index: usize,
}

#[derive(Deserialize)]
pub struct Accessor {
    #[serde(rename = "bufferView")]
    pub buffer_view_index: usize,
    #[serde(rename = "componentType")]
    pub data_type: u16,
    #[serde(rename = "type")]
    pub data_struct_type: String,
    #[serde(rename = "count")]
    pub data_struct_count: usize,
}

#[derive(Deserialize)]
pub struct BufferView {
    #[serde(rename = "buffer")]
    pub buffer_index: usize,
    #[serde(rename = "byteLength")]
    pub byte_length: usize,
    #[serde(rename = "byteOffset")]
    pub byte_offset: u64,
}

#[derive(Deserialize)]
pub struct Buffer {
    #[serde(rename = "byteLength")]
    pub byte_length: u64,
    pub uri: String,
}
