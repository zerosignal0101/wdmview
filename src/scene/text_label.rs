use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TextLabel {
    pub content: String,
    pub radius_scale: f32,
    pub position: [f32; 2],
}