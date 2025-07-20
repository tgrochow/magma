use glam::Mat4;
use glam::Vec3;

pub struct Camera {
    pub proj: Mat4,
    pub view: Mat4,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        Camera {
            proj: get_projection_matrix(aspect_ratio),
            view: Mat4::look_at_rh(
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        }
    }

    // Must be called if aspect ratio of window was changed.
    pub fn update_projection(&mut self, aspect_ratio: f32) {
        self.proj = get_projection_matrix(aspect_ratio);
    }
}

fn get_projection_matrix(aspect_ratio: f32) -> Mat4 {
    Mat4::perspective_rh_gl(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.01, 100.0)
}
