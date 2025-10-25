use std::collections::HashMap;
use std::f32::EPSILON;
use bevy_color::{Color, ColorToComponents, LinearRgba, Oklcha, Srgba};
use glam::Vec2;
use itertools::Itertools;
use wgpu::util::DeviceExt;

use crate::scene::defrag_event::AnyEvent;
use crate::scene::network::FullTopologyData;
use crate::scene::element::ElementData;
use crate::scene::connection::ConnectionData;
use crate::scene::service::ServiceData;
use crate::app_state::{State, BASE_NODE_RADIUS};
use crate::models::{Vertex2D, CircleInstance, LineVertex};


#[allow(unused)]
#[derive(Debug)]
pub enum UserCommand {
    SetFullTopology {
        elements: Vec<ElementData>,
        connections: Vec<ConnectionData>,
        defrag_timeline_events: Vec<AnyEvent>,
    },
    StateInitialized, // Notifies App that State setup is complete
    SetTimeSelection(f32), // 新增：设置时间轴选中的时刻
    SetHighlightDefragService(i32),
}

impl State {
  pub fn process_command(&mut self, command: UserCommand) {
        match command {
            UserCommand::SetFullTopology { elements, connections, defrag_timeline_events } => {
                log::info!("Setting full topology with {} nodes, {} links, and {} events.",
                            elements.len(), connections.len(), defrag_timeline_events.len());

                // 清空并重新构建节点 ID 到索引的映射
                self.node_id_to_idx.clear();
                self.node_id_to_idx = elements
                    .iter()
                    .enumerate()
                    .map(|(i, element)| (element.element_id.clone(), i))
                    .collect();

                // 存储所有节点数据
                self.all_elements = elements;
                // 存储所有连接数据
                self.all_connections = connections;
                // 存储所有服务数据
                self.all_events = defrag_timeline_events;
                
                // 根据节点数据创建圆形实例
                let radius_inside = BASE_NODE_RADIUS;
                self.circle_instances = self.all_elements
                    .iter()
                    .map(|element| CircleInstance {
                        position: [element.metadata.location.x, -element.metadata.location.y],
                        radius_scale: radius_inside + 0.2,
                        color: LinearRgba::from(Srgba::rgb_u8(0x00, 0x5d, 0x5d)).to_f32_array(),
                    })
                    .collect();

                self.line_vertices.clear();

                // 由于拓扑数据已更新，需要重新计算所有线条（服务线条将基于默认时间0）
                self.topology_needs_update = true;
                self.fit_view_to_topology();
            }
            UserCommand::StateInitialized => {
                // This command is handled in App::user_event
            }
            UserCommand::SetTimeSelection(time) => {
                // 只有当时间有实际变化时才更新，避免不必要的重新计算
                if (self.current_time_selection - time).abs() > f32::EPSILON {
                    self.current_time_selection = time;
                    self.topology_needs_update = true; // 标记服务线路需要更新
                    log::debug!("Time selection updated to: {}", time);
                }
            }
            UserCommand::SetHighlightDefragService(selected_service_id) => {
                let mut reallocation_service_id_vec = Vec::new();
                let defrag_search_result: Option<&ServiceData> = self.all_events.iter().find_map(|event| {
                    match event {
                        AnyEvent::Allocation { service_id, details, .. } => {
                            (*service_id == selected_service_id).then_some(details)
                        }
                        AnyEvent::Reallocation { timestamp, service_id, details } => {
                            if selected_service_id == details.defrag_service_id {
                                reallocation_service_id_vec.push(*service_id);
                            }
                            None
                        }
                        _ => {
                            None
                        }
                    }
                });

                match defrag_search_result {
                    Some(service_data) => {
                        log::info!(
                            "Reallocation Service IDs: {}",
                            reallocation_service_id_vec
                                .iter()
                                .map(|id| format!("{}", id))
                                .join(", ")
                        );
                        self.current_time_selection = service_data.arrival_time - EPSILON;
                        self.topology_needs_update = true;
                    }
                    None => {
                        return;
                    }
                }
            }
        }
    }
}
