use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::scene::network::FullTopologyData;
use crate::scene::node::NodeData;
use crate::scene::link::LinkData;
use crate::app_state::State;
use crate::models::{Vertex2D, CircleInstance, LineVertex};
use crate::camera::{Camera, CameraUniform};
use crate::color::Color;


#[allow(unused)]
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

impl State {
  pub fn process_command(&mut self, command: UserCommand) {
        match command {
            UserCommand::SetFullTopology { nodes, links } => {
                log::info!("Setting full topology with {} nodes and {} links.", nodes.len(), links.len());

                let node_id_to_idx: HashMap<u32, usize> = nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| (node.id, i))
                    .collect();

                self.circle_instances = nodes
                    .into_iter()
                    .map(|node| CircleInstance {
                        position: node.position.into(),
                        radius_scale: node.radius_scale,
                        color: Color::from((node.color[0], node.color[1], node.color[2])).into_linear_rgba(),
                    })
                    .collect();

                self.line_vertices.clear();
                for link in links {
                    if let (Some(&source_idx), Some(&target_idx)) = (
                        node_id_to_idx.get(&link.source_id),
                        node_id_to_idx.get(&link.target_id),
                    ) {
                        let line_color = Color::from((link.color[0], link.color[1], link.color[2]));
                        self.line_vertices.push(LineVertex {
                            position: self.circle_instances[source_idx].position,
                            color: line_color.into_linear_rgba(),
                        });
                        self.line_vertices.push(LineVertex {
                            position: self.circle_instances[target_idx].position,
                            color: line_color.into_linear_rgba(),
                        });
                    } else {
                        log::warn!("Link references non-existent node ID. Source: {}, Target: {}", link.source_id, link.target_id);
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