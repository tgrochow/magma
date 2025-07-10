pub mod mesh_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader/vert.glsl",
    }
}

pub mod mesh_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader/frag.glsl",
    }
}
