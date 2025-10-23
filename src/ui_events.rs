// src/ui_events.rs
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
                log::info!("Setting full topology with {} nodes, {} links, and {} services.",
                            elements.len(), connections.len(), services.len());

                let node_id_to_idx: HashMap<String, usize> = elements
                    .iter()
                    .enumerate()
                    .map(|(i, element)| (element.element_id.clone(), i))
                    .collect();

                let radius_inside = 20.0;

                self.circle_instances = elements
                    .into_iter()
                    .map(|element| CircleInstance {
                        position: [element.metadata.location.x, -element.metadata.location.y],
                        radius_scale: radius_inside + 0.2,
                        color: LinearRgba::from(Srgba::rgb_u8(0x00, 0x5d, 0x5d)).to_f32_array(),
                    })
                    .collect();

                self.line_vertices.clear(); // Clear all previous lines

                // The 'rotate_angle' defines the maximal angular offset from the center line
                // for the link boundaries.
                const LINK_BOUNDARY_ROTATE_ANGLE: f32 = std::f32::consts::PI / 16.0; // Approximately 0.201 radians
                // --- Render Links (existing logic for link boundaries) ---
                for link in connections {
                    if let (Some(&source_idx), Some(&target_idx)) = (
                        node_id_to_idx.get(&link.from_node),
                        node_id_to_idx.get(&link.to_node),
                    ) {
                        let link_boundary_color = LinearRgba::from(Srgba::rgb_u8(230, 230, 230));
                        let source_position_center = Vec2::from_array(self.circle_instances[source_idx].position);
                        let destination_position_center = Vec2::from_array(self.circle_instances[target_idx].position);
                        let dir_vec = destination_position_center - source_position_center;
                        let length = dir_vec.length();

                        // Avoid division by zero or rendering extremely short segments
                        if length < f32::EPSILON {
                            continue;
                        }

                        let normalized_dir = dir_vec.normalize();
                        let radius_dir_outward = normalized_dir * radius_inside; // Vector from source center to circumference, outward

                        let rotate_vector = Vec2::from_angle(LINK_BOUNDARY_ROTATE_ANGLE);
                        let reverse_rotate_vector = Vec2::from_angle(-LINK_BOUNDARY_ROTATE_ANGLE);

                        // Upper boundary line
                        self.line_vertices.push(LineVertex {
                            position: (source_position_center + radius_dir_outward.rotate(rotate_vector)).into(),
                            color: link_boundary_color.to_f32_array(),
                        });
                        self.line_vertices.push(LineVertex {
                            // Target point: from target center, move inward along dir_vec, then rotate
                            position: (destination_position_center - radius_dir_outward.rotate(reverse_rotate_vector)).into(),
                            color: link_boundary_color.to_f32_array(),
                        });

                        // Lower boundary line
                        self.line_vertices.push(LineVertex {
                            position: (source_position_center + radius_dir_outward.rotate(reverse_rotate_vector)).into(),
                            color: link_boundary_color.to_f32_array(),
                        });
                        self.line_vertices.push(LineVertex {
                            position: (destination_position_center - radius_dir_outward.rotate(rotate_vector)).into(),
                            color: link_boundary_color.to_f32_array(),
                        });
                    } else {
                        log::warn!("Link references non-existent node ID. Source: {}, Target: {}", link.from_node, link.to_node);
                    }
                }

                // --- Render Services ---
                const MAX_WAVELENGTHS: u32 = 80; // Fixed number of carrier wavelengths
                // Services will be rendered within the boundaries of the links.
                // SERVICE_MAX_SPREAD_ANGLE defines the maximum angular deviation from the
                // central link axis for the outermost service wavelength.
                // It should be slightly less than LINK_BOUNDARY_ROTATE_ANGLE to ensure
                // service lines are visually "inside" the link boundaries.
                const SERVICE_MAX_SPREAD_ANGLE: f32 = LINK_BOUNDARY_ROTATE_ANGLE * 0.95; // e.g., 95% of the link boundary spread

                for service in services {
                    let wavelength = service.wavelength;
                    if wavelength >= MAX_WAVELENGTHS {
                        log::warn!(
                            "Service {} uses wavelength {} which is outside MAX_WAVELENGTHS ({}) - clamping to MAX_WAVELENGTHS - 1.",
                            service.service_id, wavelength, MAX_WAVELENGTHS
                        );
                        // Clamp wavelength to stay within bounds for consistent color/offset calculation
                        // MAX_WAVELENGTHS - 1 is the highest valid index (79 for 80 wavelengths).
                        // `min` is safe since wavelength is `u32`.
                        // Casting entire expression to f32 at once is more robust
                        // (wavelength as f32) will be compared with (MAX_WAVELENGTHS - 1) as f32
                        // if wavelength=80.0, will be 79.0
                        // if wavelength=79.0, will be 79.0
                    }
                    let effective_wavelength = (wavelength as f32).min((MAX_WAVELENGTHS - 1) as f32);

                    // Calculate hue color based on wavelength. Formula from user example.
                    // This distributes hues from Oklcha's LCH color space across the spectrum.
                    let hue_color = (effective_wavelength + 0.5) / (MAX_WAVELENGTHS as f32) * 180.0 + 30.0;
                    let service_color_oklcha = Oklcha::lch(0.7289, 0.11, hue_color);
                    let service_color_f32 = LinearRgba::from(service_color_oklcha).to_f32_array();

                    // Calculate angular offset for this particular wavelength.
                    // This creates a normalized factor from -1.0 (lowest wavelength) to +1.0 (highest wavelength).
                    let normalized_wavelength_factor = (effective_wavelength - ((MAX_WAVELENGTHS as f32 - 1.0) / 2.0)) / ((MAX_WAVELENGTHS as f32 - 1.0) / 2.0);
                    // Apply this factor to the maximum spread angle to get the actual rotation for this wavelength.
                    let wavelength_rotate_angle = normalized_wavelength_factor * SERVICE_MAX_SPREAD_ANGLE;

                    // Iterate through each segment of the service's path
                    for i in 0..(service.path.len() - 1) {
                        let source_node_id = &service.path[i];
                        let target_node_id = &service.path[i + 1];

                        if let (Some(&source_idx), Some(&target_idx)) = (
                            node_id_to_idx.get(source_node_id),
                            node_id_to_idx.get(target_node_id),
                        ) {
                            let source_pos_center = Vec2::from_array(self.circle_instances[source_idx].position);
                            let target_pos_center = Vec2::from_array(self.circle_instances[target_idx].position);

                            let dir_vec = target_pos_center - source_pos_center;
                            let length = dir_vec.length();

                            if length < f32::EPSILON {
                                continue; // Skip very short segments
                            }

                            let normalized_dir = dir_vec.normalize();
                            // This vector points from source center towards target center, scaled by node radius.
                            let radius_vec_along_link = normalized_dir * radius_inside;

                            // Calculate the exact start and end points on the circumference for this service's wavelength.
                            let upward_sacle: f32 = match normalized_dir.y >= 0.0 {
                                true => 1.0,
                                false => -1.0,
                            };
                            let service_start_pos = source_pos_center + radius_vec_along_link.rotate(Vec2::from_angle(wavelength_rotate_angle * upward_sacle));
                            let service_end_pos = target_pos_center - radius_vec_along_link.rotate(Vec2::from_angle( - wavelength_rotate_angle * upward_sacle));

                            self.line_vertices.push(LineVertex {
                                position: service_start_pos.into(),
                                color: service_color_f32,
                            });
                            self.line_vertices.push(LineVertex {
                                position: service_end_pos.into(),
                                color: service_color_f32,
                            });
                        } else {
                            log::warn!(
                                "Service {} path references non-existent node ID. Segment: {} -> {}",
                                service.service_id, source_node_id, target_node_id
                            );
                        }
                    } // End of path segments loop

                    // Processing the segments inside the circle
                    for i in 0..(service.path.len() - 2) {
                        let source_node_id = &service.path[i];
                        let middle_node_id = &service.path[i + 1];
                        let target_node_id = &service.path[i + 2];

                        if let (Some(&source_idx), Some(&middle_idx), Some(&target_idx)) = (
                            node_id_to_idx.get(source_node_id),
                            node_id_to_idx.get(middle_node_id),
                            node_id_to_idx.get(target_node_id),
                        ) {
                            let source_pos_center = Vec2::from_array(self.circle_instances[source_idx].position);
                            let middle_pos_center = Vec2::from_array(self.circle_instances[middle_idx].position);
                            let target_pos_center = Vec2::from_array(self.circle_instances[target_idx].position);

                            let source_middle_dir_vec = target_pos_center - middle_pos_center;
                            let middle_target_dir_vec = middle_pos_center - source_pos_center;

                            let normalized_source_middle_dir = source_middle_dir_vec.normalize();
                            let normalized_middle_target_dir = middle_target_dir_vec.normalize();
                            // This vector points from source center towards target center, scaled by node radius.
                            let radius_source_middle_vec_along_link = normalized_source_middle_dir * radius_inside;
                            let radius_middle_target_vec_along_link = normalized_middle_target_dir * radius_inside;

                            // Calculate the exact start and end points on the circumference for this service's wavelength.
                            let source_middle_upward_sacle: f32 = match normalized_source_middle_dir.y >= 0.0 {
                                true => 1.0,
                                false => -1.0,
                            };
                            let middle_target_upward_sacle: f32 = match normalized_middle_target_dir.y >= 0.0 {
                                true => 1.0,
                                false => -1.0,
                            };
                            let middle_start_pos = middle_pos_center + radius_source_middle_vec_along_link.rotate(Vec2::from_angle(wavelength_rotate_angle * source_middle_upward_sacle));
                            let middle_end_pos = middle_pos_center - radius_middle_target_vec_along_link.rotate(Vec2::from_angle( - wavelength_rotate_angle * middle_target_upward_sacle));

                            self.line_vertices.push(LineVertex {
                                position: middle_start_pos.into(),
                                color: service_color_f32,
                            });
                            self.line_vertices.push(LineVertex {
                                position: middle_end_pos.into(),
                                color: service_color_f32,
                            });
                        } else {
                            log::warn!(
                                "Service {} path references non-existent node ID. Segment: {} -> {}",
                                service.service_id, source_node_id, target_node_id
                            );
                        }
                    } // End of inside segments
                } // End of services loop

                self.update_gpu_buffers(); // Update GPU buffers after all line vertices (links + services) are added
                self.fit_view_to_topology();
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

      // (Re)create circle instance buffer if size changes, otherwise write
      if self.circle_instance_buffer.size() < circle_data.len() as u64 {
          self.circle_instance_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
              label: Some("Circle Instance Buffer (Resized)"),
              contents: circle_data,
              usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
          });
      } else {
          self.queue.write_buffer(&self.circle_instance_buffer, 0, circle_data);
      }

      // (Re)create line vertex buffer if size changes, otherwise write
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
