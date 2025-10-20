use serde::Deserialize;

use crate::scene::service::ServiceData;

use super::element::ElementData;
use super::connection::LinkData;

#[derive(Deserialize, Debug)]
pub struct FullTopologyData {
    pub elements: Vec<ElementData>,
    pub connections: Vec<LinkData>,
    pub services: Vec<ServiceData>,
}