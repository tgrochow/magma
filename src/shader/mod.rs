pub mod mesh_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader/glsl/vert.glsl",
    }
}

pub mod pbr_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader/glsl/vert_pbr.glsl",
    }
}

pub mod mesh_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader/glsl/frag.glsl",
    }
}

pub mod debug_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader/glsl/debug.glsl",
    }
}

pub mod pbr_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader/glsl/frag_pbr.glsl",
    }
}
