use serde::Deserialize;
use super::service::ServiceData;
use std::collections::HashMap;

// Using derive macros to automatically implement features.
// - Deserialize: Allows creating this struct from data formats like JSON.
// - Debug: Allows printing the struct for debugging using `{:?}`.
// - Clone: Allows making copies of the struct, which is useful for our state map.
#[derive(Deserialize, Debug, Clone)]
pub struct DefragResult {
    blocknum1: i32,
    blocknum2: i32,
}

// ReallocationDetails "inherits" DefragService in Python.
// In Rust, we represent this with composition. The #[serde(flatten)] attribute
// tells serde to pull all fields from the `service` field into this struct during
// deserialization, making it look like a single flat object in the JSON.
#[derive(Deserialize, Debug, Clone)]
pub struct ReallocationDetails {
    pub defrag_service_id: i32,
    #[serde(flatten)]
    pub service: ServiceData,
}

// We need a way to get a DefragService from ReallocationDetails for our state map.
// Implementing the `From` trait is the idiomatic Rust way to do this conversion.
impl From<ReallocationDetails> for ServiceData {
    fn from(details: ReallocationDetails) -> Self {
        details.service
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct ReleaseExpiredDetails {
    pub departure_time: f32,
}

// This is the core of the solution. It mirrors the Pydantic tagged union.
// #[serde(tag = "event_type")] tells serde to look at the "event_type" field
// in the JSON to decide which enum variant to create.
// #[serde(rename_all = "SCREAMING_SNAKE_CASE")] handles the naming convention
// (e.g., "ALLOCATION" in JSON maps to the `Allocation` variant in Rust).
#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "event_type")]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AnyEvent {
    Allocation {
        timestamp: f32,
        service_id: i32,
        // In this case, AllocationDetails is just a DefragService
        details: ServiceData,
    },
    ReleaseExpired {
        timestamp: f32,
        service_id: i32,
        details: ReleaseExpiredDetails,
    },
    // Note: Python adds a 'RELEASE_DEFRAG' case. We can add it easily here if needed.
    // ReleaseDefrag {
    //     timestamp: f32,
    //     service_id: i32,
    // },
    Reallocation {
        timestamp: f32,
        service_id: i32,
        details: ReallocationDetails,
    },
}

// Helper function to get the timestamp from any event variant without
// needing to write a full match statement every time.
impl AnyEvent {
    pub fn timestamp(&self) -> f32 {
        match self {
            AnyEvent::Allocation { timestamp, .. } => *timestamp,
            AnyEvent::ReleaseExpired { timestamp, .. } => *timestamp,
            AnyEvent::Reallocation { timestamp, .. } => *timestamp,
        }
    }
}


#[derive(Deserialize, Debug)]
pub struct DefragResponse {
    pub result: DefragResult,
    pub defrag_timeline_events: Vec<AnyEvent>,
}

/// Reconstructs the service dictionary state at a specific target time
/// by replaying events from a timeline.
pub fn reconstruct_state_at_time(
    timeline_events: &[AnyEvent],
    target_time: f32,
) -> HashMap<i32, ServiceData> {
    // The Python code sorts the events. Assuming they might not be sorted,
    // we can do that here. If they are guaranteed to be sorted, this can be skipped.
    // Note: This requires cloning the events. A more efficient way would be to sort
    // a vector of indices, but for clarity, we'll clone.
    // let mut sorted_events: Vec<AnyEvent> = timeline_events.to_vec();
    // sorted_events.sort_by(|a, b| a.timestamp().partial_cmp(&b.timestamp()).unwrap());

    // We initialize our state map. The key is the service ID.
    let mut reconstructed_service_dict: HashMap<i32, ServiceData> = HashMap::new();

    // The Python example assumes events are pre-sorted, so we will too for efficiency.
    // Iterate over the events.
    for event in timeline_events { // If sorting, iterate over `&sorted_events`
        // Only process events that occurred at or before the target time.
        if event.timestamp() > target_time {
            break;
        }

        // Use a `match` statement to handle each event type.
        // This is safer and more expressive than if/elif string checks.
        match event {
            AnyEvent::Allocation { service_id, details, .. } => {
                // Insert the new service into our state map.
                // We clone `details` because the map takes ownership.
                reconstructed_service_dict.insert(*service_id, details.clone());
            }
            AnyEvent::ReleaseExpired { service_id, .. } => {
                // Remove the service from the map.
                reconstructed_service_dict.remove(service_id);
            }
            // Add a case for `RELEASE_DEFRAG` if you model it in the enum
            // AnyEvent::ReleaseDefrag { service_id, .. } => {
            //     reconstructed_service_dict.remove(service_id);
            // }
            AnyEvent::Reallocation { service_id, details, .. } => {
                // Convert the ReallocationDetails into a DefragService using our
                // `From` implementation and update the map.
                let updated_service: ServiceData = details.clone().into();
                reconstructed_service_dict.insert(*service_id, updated_service);
            }
        }
    }

    reconstructed_service_dict
}
