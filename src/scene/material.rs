use crate::scene::gltf;
use vulkano::padded::Padded;

#[derive(Clone)]
pub struct Material {
    pub use_color_texture: Padded<u32, 12>,
    pub color: [f32; 4],
    pub color_texture_index: usize,
    pub mettalic: f32,
    pub roughness: f32,
}

impl Material {
    pub fn new(m: &gltf::Material) -> Self {
        let use_color_texture = Padded(m.pbr.color_texture.map_or(0, |_| 1));
        let color_texture_index = m
            .pbr
            .color_texture
            .map(|texture| texture.texture_index)
            .unwrap_or(0);
        Material {
            use_color_texture: use_color_texture,
            color: m.pbr.color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            color_texture_index: color_texture_index,
            mettalic: m.pbr.mettalic.unwrap_or(0.0),
            roughness: m.pbr.roughness.unwrap_or(0.0),
        }
    }
}
