use crate::engine::model::Model;

use std::collections::HashMap;

pub struct Scene {
    pub models: HashMap<String, Model>,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            models: HashMap::new(),
        }
    }
}
