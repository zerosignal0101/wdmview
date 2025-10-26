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
    AttachCanvas(String),
    SetFullTopology {
        elements: Vec<ElementData>,
        connections: Vec<ConnectionData>,
        defrag_timeline_events: Vec<AnyEvent>,
    },
    StateInitialized, // Notifies App that State setup is complete
    SetTimeSelection(f32), // 新增：设置时间轴选中的时刻
    SetHighlightDefragService(i32),
    DestroyView,
}

impl State {
    pub fn process_command(&mut self, command: UserCommand) {
        match command {
            UserCommand::SetFullTopology { elements, connections, defrag_timeline_events } => {
                log::info!("Setting full topology with {} nodes, {} links, and {} events.",
                            elements.len(), connections.len(), defrag_timeline_events.len());

                self.node_id_to_idx.clear();
                self.node_id_to_idx = elements
                    .iter()
                    .enumerate()
                    .map(|(i, element)| (element.element_id.clone(), i))
                    .collect();

                self.all_elements = elements;
                self.all_connections = connections;
                self.all_events = defrag_timeline_events;
                
                // 初始化（或重置）所有节点的默认颜色
                let default_node_color = LinearRgba::from(Srgba::rgb_u8(0x00, 0x5d, 0x5d)).to_f32_array();
                self.circle_instances = self.all_elements
                    .iter()
                    .map(|element| CircleInstance {
                        position: [element.metadata.location.x, -element.metadata.location.y],
                        radius_scale: BASE_NODE_RADIUS + 0.2, // 初始半径
                        color: default_node_color, // 初始颜色
                    })
                    .collect();

                self.line_vertices.clear();
                self.highlight_line_vertices.clear(); // 清空高亮线条

                self.topology_needs_update = true;
                self.current_time_selection = 0.0; // Reset time to 0
                self.highlight_service_id_list = None; // Clear highlight
                self.fit_view_to_topology();
            }
            UserCommand::StateInitialized => {
                // ...
            }
            UserCommand::AttachCanvas (_) => {
                // ...
            }
            UserCommand::DestroyView => {
                // ...
            }
            UserCommand::SetTimeSelection(time) => {
                if (self.current_time_selection - time).abs() > f32::EPSILON {
                    self.current_time_selection = time;
                    self.highlight_service_id_list = None; // 清除高亮服务
                    self.topology_needs_update = true;
                    log::debug!("Time selection updated to: {}", time);
                }
            }
            UserCommand::SetHighlightDefragService(selected_service_id) => {
                let mut highlight_service_id_vec = Vec::new();
                let mut arrival_time_for_highlight = 0.0;
                let mut found_service = false;

                // 遍历所有事件，找出与 selected_service_id 相关的所有服务ID
                for event in &self.all_events {
                    match event {
                        AnyEvent::Allocation { service_id, details, .. } => {
                            if selected_service_id == *service_id {
                                highlight_service_id_vec.push(*service_id);
                                arrival_time_for_highlight = details.arrival_time;
                                found_service = true;
                            }
                        }
                        AnyEvent::Reallocation { service_id, details, .. } => {
                            // 如果 re-allocation 的来源是 selected_service_id
                            if selected_service_id == details.defrag_service_id {
                                // highlight_service_id_vec.push(*service_id);
                                // 如果主要服务还没找到，则将此 reallocation 的 arrival_time 作为时间起点
                                if !found_service {
                                    arrival_time_for_highlight = details.service.arrival_time;
                                    // 注意：这里可能需要更复杂的逻辑来确定一个合理的起始时间，
                                    // 比如找到所有相关服务中最早的 arrival_time
                                    found_service = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }

                if found_service && !highlight_service_id_vec.is_empty() {
                    log::info!(
                        "Highlight Service IDs: {:?}",
                        highlight_service_id_vec,
                    );
                    // 将时间设置到找到服务的开始时间，稍微加一点 EPSILON 确保在活跃期内
                    self.current_time_selection = arrival_time_for_highlight + EPSILON;
                    self.highlight_service_id_list = Some(highlight_service_id_vec);
                    self.topology_needs_update = true; // 标记需要更新拓扑以显示高亮
                    self.fit_view_to_topology(); // 可能需要重新调整视角
                } else {
                    log::warn!("Service ID {} not found or is not a defragmentation service.", selected_service_id);
                    self.highlight_service_id_list = None; // 确保清除高亮
                    self.topology_needs_update = true;
                }
            }
        }
    }
}
