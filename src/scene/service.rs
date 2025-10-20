use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServiceData {
    pub service_id: u64,
    pub source_id: String,
    pub destination_id: String,
    pub arrival_time: f32,
    pub holding_time: f32,
    pub bit_rate: f32,
    pub modulation: Option<String>, // 使用 Option<String> 因为 JSON 中是 "modulation": null
    pub power: f32,
    pub path: Vec<String>,
    pub wavelength: u32,
    pub snr_requirement: f32,
    #[serde(rename = "GSNR")]
    pub gsnr: f32,
    pub utilization: f32,
}