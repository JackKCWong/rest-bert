use std::sync::{Arc, Mutex};
use async_object_pool::Pool;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;

pub struct AppState {
    pub models: Pool<Arc<Mutex<SentenceEmbeddingsModel>>>,
}
