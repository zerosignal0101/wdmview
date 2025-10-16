use serde::Deserialize;

use super::node::NodeData;
use super::link::LinkData;

#[derive(Deserialize, Debug)]
pub struct FullTopologyData {
    pub nodes: Vec<NodeData>,
    pub links: Vec<LinkData>,
}