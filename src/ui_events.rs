use std::collections::HashMap;
use bevy_color::{Color, ColorToComponents, LinearRgba, Oklcha, Srgba};
use glam::Vec2;
use wgpu::util::DeviceExt;

use crate::scene::network::FullTopologyData;
use crate::scene::element::ElementData;
use crate::scene::connection::LinkData;
use crate::scene::service::ServiceData;
use crate::app_state::{State, BASE_NODE_RADIUS};
use crate::models::{Vertex2D, CircleInstance, LineVertex};


#[allow(unused)]
#[derive(Debug)]
pub enum UserCommand {
    SetFullTopology {
        elements: Vec<ElementData>,
        connections: Vec<LinkData>,
        services: Vec<ServiceData>,
    },
    AddNode(ElementData),
    RemoveNode(u32),
    StateInitialized, // Notifies App that State setup is complete
    SetTimeSelection(f32), // 新增：设置时间轴选中的时刻
}

impl State {
  pub fn process_command(&mut self, command: UserCommand) {
        match command {
            UserCommand::SetFullTopology { elements, connections, services } => {
                log::info!("Setting full topology with {} nodes, {} links, and {} services.",
                            elements.len(), connections.len(), services.len());

                // 清空并重新构建节点 ID 到索引的映射
                self.node_id_to_idx.clear();
                self.node_id_to_idx = elements
                    .iter()
                    .enumerate()
                    .map(|(i, element)| (element.element_id.clone(), i))
                    .collect();

                // 存储所有节点数据
                self.all_elements = elements;
                // 存储所有服务数据
                self.all_services = services;
                
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

                // 清除旧的线条数据，以便重新计算所有线条（包括静态链路和动态服务）
                // 链路边界线也是静态的，如果 connections 是固定不变的，可以在这里绘制一次
                // 并存储到单独的 `link_lines: Vec<LineVertex>` 中，后续时间改变时只更新服务线。
                // 为了简化，这里暂时在 `generate_all_lines_for_current_time` 中统一处理 (但效率会低一点)
                // 或者，我们可以确保 connections 的线条只在这里生成一次并写入 `line_vertices`，
                // 然后 services 的线条逻辑可以追加（需要更精细的buffer管理或在State里区分）。
                // 为了避免 `generate_all_lines_for_current_time` 依赖 connections，我们先假定 connections 产生的数据
                // 也是通过服务路径隐含的，或者在 time_selection 之前已经存在。
                // 更好的解决方案是：
                // 1. `State` 中增加 `pub static_link_line_vertices: Vec<LineVertex>`
                // 2. `SetFullTopology` 时，用 `connections` 填充 `static_link_line_vertices`
                // 3. `generate_all_lines_for_current_time` 重新构建 `line_vertices` 时，
                //    先 `self.line_vertices.extend_from_slice(&self.static_link_line_vertices);`
                //    然后再添加活跃的服务线条。
                // 这里暂时直接清空 `self.line_vertices` 并让 `generate_all_lines_for_current_time` 负责所有线条绘制。
                self.line_vertices.clear();

                // 由于拓扑数据已更新，需要重新计算所有线条（服务线条将基于默认时间0）
                self.topology_needs_update = true;
                self.fit_view_to_topology();
            }
            UserCommand::AddNode(node_data) => {
                log::info!("Add node command received: {:?}", node_data);
                // 这里您可以根据需要实现添加节点的逻辑，
                // 并记得更新 `self.all_elements`, `self.node_id_to_idx`, `self.circle_instances`
                // 并且将 `topology_needs_update` 设为 true，以便重新绘制。
            }
            UserCommand::RemoveNode(node_id) => {
                log::info!("Remove node command received: {:?}", node_id);
                // 同上，实现移除节点的逻辑，更新相关数据结构，并设置 `topology_needs_update = true`。
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
        }
    }
}
