use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServiceData {
    pub source_id: String,
    pub destination_id: String,
    pub arrival_time: f32,
    pub departure_time: f32,
    pub snr_requirement: f32,
    pub bit_rate_requirement: f32,
    pub power: f32,
    pub path: Vec<String>,
    pub wavelength: i32,
    pub gsnr: f32,
    pub utilization: f32,
}