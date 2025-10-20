use serde::{Deserialize, Serialize};

/// 表示地理位置的坐标
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Location {
    pub x: f32,
    pub y: f32,
}

/// 表示节点的元数据，其中包含位置信息
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Metadata {
    pub location: Location,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ElementData {
    pub name: String,
    
    #[serde(rename = "type")]
    pub node_type: String, // 使用一个非关键字段的名称，并映射到 JSON 的 "type"
    
    pub type_variety: String,
    pub metadata: Metadata, // 这里直接使用 Metadata 结构体
    pub element_id: String,
}