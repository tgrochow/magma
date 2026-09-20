use glam::{Mat4, Vec3};

pub struct Camera {
    proj: Mat4,
    view: Mat4,
    pos: Vec3,
    dir: Vec3,
    up: Vec3,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        let pos = Vec3::new(0.0, 3.0, 6.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let up = Vec3::new(0.0, -1.0, 0.0);
        Self {
            pos: pos,
            dir: dir,
            up: up,
            proj: get_projection_matrix(aspect_ratio),
            view: get_view_matrix(pos, dir, up),
        }
    }

    pub fn get_position(&self) -> Vec3 {
        return self.pos.clone();
    }

    pub fn get_projection_matrix(&self) -> Mat4 {
        self.proj.clone()
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        self.view.clone()
    }

    // must be called if aspect ratio of window was changed
    pub fn update_projection(&mut self, aspect_ratio: f32) {
        self.proj = get_projection_matrix(aspect_ratio);
    }

    fn update_view(&mut self) {
        self.view = get_view_matrix(self.pos, self.dir, self.up)
    }

    pub fn move_forward(&mut self) {
        let forward = self.dir * 0.1;
        self.pos += forward;
        self.update_view()
    }

    pub fn move_backwards(&mut self) {
        let backwards = self.dir * -0.1;
        self.pos += backwards;
        self.update_view()
    }
}

fn get_view_matrix(pos: Vec3, dir: Vec3, up: Vec3) -> Mat4 {
    glam::camera::rh::view::look_at_mat4(pos, pos + dir, up)
}

fn get_projection_matrix(aspect_ratio: f32) -> Mat4 {
    glam::camera::rh::proj::opengl::perspective(
        std::f32::consts::FRAC_PI_2,
        aspect_ratio,
        0.01,
        100.0,
    )
}
