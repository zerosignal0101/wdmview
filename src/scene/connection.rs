use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConnectionData {
    pub from_node: String,
    pub to_node: String,
    pub connection_id: String,
}