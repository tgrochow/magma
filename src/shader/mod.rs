pub mod simple_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader/glsl/simple.vert",
    }
}

pub mod pbr_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader/glsl/pbr.vert",
    }
}

pub mod simple_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader/glsl/simple.frag",
    }
}

pub mod debug_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader/glsl/debug.frag",
    }
}

pub mod pbr_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader/glsl/pbr.frag",
    }
}
