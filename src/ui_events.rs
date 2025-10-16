use crate::scene::network::FullTopologyData;
use crate::scene::node::NodeData;
use crate::scene::link::LinkData;

#[derive(Debug)]
pub enum UserCommand {
    SetFullTopology {
        nodes: Vec<NodeData>,
        links: Vec<LinkData>,
    },
    AddNode(NodeData),
    RemoveNode(u32),
    StateInitialized, // Notifies App that State setup is complete
}
