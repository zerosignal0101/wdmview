use serde::Deserialize;
use super::service::ServiceData;
use std::collections::HashMap;


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


/// Reconstructs the service dictionary state at a specific target time
/// by replaying events from a timeline.
pub fn reconstruct_state_at_time(
    timeline_events: &[AnyEvent],
    target_time: f32,
) -> HashMap<i32, ServiceData> {
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
