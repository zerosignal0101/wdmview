use std::collections::HashMap;
use bevy_color::{Color, ColorToComponents, LinearRgba, Oklcha, Srgba};
use glam::Vec2;
use wgpu::util::DeviceExt;

use crate::scene::network::FullTopologyData;
use crate::scene::element::ElementData;
use crate::scene::connection::LinkData;
use crate::scene::service::ServiceData;
use crate::app_state::State;
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
}

impl State {
  pub fn process_command(&mut self, command: UserCommand) {
        match command {
            UserCommand::SetFullTopology { elements, connections, services } => {
                log::info!("Setting full topology with {} nodes and {} links.", elements.len(), connections.len());

                let node_id_to_idx: HashMap<String, usize> = elements
                    .iter()
                    .enumerate()
                    .map(|(i, element)| (element.element_id.clone(), i))
                    .collect();

                self.circle_instances = elements
                    .into_iter()
                    .map(|element| CircleInstance {
                        position: [element.metadata.location.x, element.metadata.location.y],
                        radius_scale: 25.0,
                        color: LinearRgba::from(Srgba::rgb_u8(0x00, 0x5d, 0x5d)).to_f32_array(),
                    })
                    .collect();

                self.line_vertices.clear();
                for link in connections {
                    if let (Some(&source_idx), Some(&target_idx)) = (
                        node_id_to_idx.get(&link.from_node),
                        node_id_to_idx.get(&link.to_node),
                    ) {
                        let line_color = LinearRgba::from(Srgba::rgb_u8(230, 230, 230));
                        let source_position = Vec2::from_array(self.circle_instances[source_idx].position);
                        let destination_position = Vec2::from_array(self.circle_instances[target_idx].position);
                        let dir_vec = destination_position - source_position;
                        let length = dir_vec.length();

                        // 避免除以零或非常短的线段
                        if length < f32::EPSILON {
                            continue;
                        }

                        let normalized_dir = dir_vec.normalize();

                        self.line_vertices.push(LineVertex {
                            position: source_position.into(),
                            color: line_color.to_f32_array(),
                        });
                        self.line_vertices.push(LineVertex {
                            position: destination_position.into(),
                            color: line_color.to_f32_array(),
                        });
                    } else {
                        log::warn!("Link references non-existent node ID. Source: {}, Target: {}", link.from_node, link.to_node);
                    }
                }

                self.update_gpu_buffers();
            }
            UserCommand::AddNode(node_data) => {
                // Implement add node logic
                log::info!("Add node command received: {:?}", node_data);
            }
            UserCommand::RemoveNode(node_id) => {
                // Implement remove node logic
                log::info!("Remove node command received: {:?}", node_id);
            }
            UserCommand::StateInitialized => {
                // This command is handled in App::user_event
            }
        }
    }

    fn update_gpu_buffers(&mut self) {
      let circle_data = bytemuck::cast_slice(&self.circle_instances);
      let line_data = bytemuck::cast_slice(&self.line_vertices);

      if self.circle_instance_buffer.size() < circle_data.len() as u64 {
          self.circle_instance_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
              label: Some("Circle Instance Buffer (Resized)"),
              contents: circle_data,
              usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
          });
      } else {
          self.queue.write_buffer(&self.circle_instance_buffer, 0, circle_data);
      }

      if self.line_vertex_buffer.size() < line_data.len() as u64 {
          self.line_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
              label: Some("Line Vertex Buffer (Resized)"),
              contents: line_data,
              usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
          });
      } else {
          self.queue.write_buffer(&self.line_vertex_buffer, 0, line_data);
      }
  }
}