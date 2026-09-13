use std::sync::Arc;
use vulkano::device::physical::PhysicalDevice;
use vulkano::image::SampleCount;

pub struct Config {
    pub sample_count: SampleCount,
}

impl Config {
    pub fn new(physical_device: Arc<PhysicalDevice>, sample_count: SampleCount) -> Self {
        let properties = physical_device.properties();
        let color_counts = properties.framebuffer_color_sample_counts;
        let depth_counts = properties.framebuffer_depth_sample_counts;
        if !color_counts.contains_enum(sample_count) || !depth_counts.contains_enum(sample_count) {
            panic!("invalid config")
        }
        Config {
            sample_count: sample_count,
        }
    }
}
