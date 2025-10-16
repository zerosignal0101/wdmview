use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinkData {
    pub source_id: u32,
    pub target_id: u32,
    pub color: [u8; 3],
}