use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeData {
    pub id: u32,
    pub position: [f32; 2],
    pub radius_scale: f32,
    pub color: [u8; 3],
}