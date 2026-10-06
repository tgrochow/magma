use serde::Deserialize;

#[derive(Deserialize)]
pub struct GLTF {
    pub materials: Vec<Material>,
    pub meshes: Vec<Mesh>,
    pub textures: Vec<Texture>,
    pub images: Vec<Image>,
    pub accessors: Vec<Accessor>,
    #[serde(rename = "bufferViews")]
    pub buffer_views: Vec<BufferView>,
    pub samplers: Vec<Sampler>,
    pub buffers: Vec<Buffer>,
}

#[derive(Clone, Deserialize)]
pub struct Material {
    #[serde(rename = "pbrMetallicRoughness")]
    pub pbr: PBR,
}

#[derive(Clone, Deserialize)]
pub struct PBR {
    #[serde(rename = "baseColorFactor")]
    pub color: Option<[f32; 4]>,
    #[serde(rename = "baseColorTexture")]
    pub color_texture: Option<PBRTexture>,
    #[serde(rename = "metallicFactor")]
    pub mettalic: Option<f32>,
    #[serde(rename = "roughnessFactor")]
    pub roughness: Option<f32>,
}

#[derive(Clone, Copy, Deserialize)]
pub struct PBRTexture {
    #[serde(rename = "index")]
    pub texture_index: usize,
}

#[derive(Deserialize)]
pub struct Mesh {
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
pub struct Texture {
    #[serde(rename = "sampler")]
    pub sampler_id: usize,
    #[serde(rename = "source")]
    pub source_id: usize,
}

#[derive(Deserialize)]
pub struct Image {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    #[serde(rename = "uri")]
    pub uri: String,
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
pub struct Sampler {
    #[serde(rename = "magFilter")]
    pub mag_filter: usize,
    #[serde(rename = "minFilter")]
    pub min_filter: usize,
}

#[derive(Deserialize)]
pub struct Buffer {
    #[serde(rename = "byteLength")]
    pub byte_length: u64,
    pub uri: String,
}
