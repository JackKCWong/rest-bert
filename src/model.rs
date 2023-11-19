use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub inputs: Vec<String>,
}

#[derive(Deserialize, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<Embedding>,
    pub model: &'static str,
}

#[derive(Deserialize, Serialize)]
pub struct Embedding {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}
