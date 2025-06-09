pub mod screen_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader/screen/vert.glsl",
    }
}

pub mod screen_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader/screen/frag.glsl",
    }
}

pub mod mesh_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader/mesh/vert.glsl",
    }
}

pub mod mesh_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader/mesh/frag.glsl",
    }
}
