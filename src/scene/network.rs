use serde::Deserialize;

use crate::scene::defrag_event::AnyEvent;

use super::element::ElementData;
use super::connection::ConnectionData;

#[derive(Deserialize, Debug)]
pub struct FullTopologyData {
    pub elements: Vec<ElementData>,
    pub connections: Vec<ConnectionData>,
    pub defrag_timeline_events: Vec<AnyEvent>,
}