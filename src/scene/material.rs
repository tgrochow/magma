use crate::scene::gltf;

#[derive(Clone)]
pub struct Material {
    pub color: [f32; 4],
    pub color_texture_index: usize,
    pub mettalic: f32,
    pub roughness: f32,
}

impl Material {
    pub fn new(m: &gltf::Material) -> Self {
        let color_texture_index = m
            .pbr
            .color_texture
            .map(|texture| texture.texture_index)
            .unwrap_or(0);
        Material {
            color: m.pbr.color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            color_texture_index: color_texture_index,
            mettalic: m.pbr.mettalic.unwrap_or(0.0),
            roughness: m.pbr.roughness.unwrap_or(0.0),
        }
    }
}
