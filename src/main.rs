use std::sync::Mutex;

use actix_web::{
    post,
    web::{self},
    App, HttpResponse, HttpServer, Responder,
};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize, Serialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<Embedding>,
    model: String,
}

#[derive(Deserialize, Serialize)]
struct Embedding {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

struct AppState {
    model: Mutex<SentenceEmbeddingsModel>,
}

#[post("/v1/embeddings")]
async fn sentence_embedding(
    app: web::Data<AppState>,
    form: web::Json<EmbeddingRequest>,
) -> impl Responder {

    let tensors = app.model.lock().unwrap().encode(&form.input);
    if tensors.is_err() {
        return HttpResponse::InternalServerError().body("Could not embed input");
    }

    let tensors = tensors.unwrap();
    let mut embeddings = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        embeddings.push(Embedding {
            object: "embedding".to_owned(),
            embedding: t.to_vec(),
            index: i,
        });
    }

    match serde_json::to_string(&EmbeddingResponse {
        model: "all-miniLM-L12-v2".to_owned(),
        object: "list".to_owned(),
        data: embeddings,
    }) {
        Ok(body) => HttpResponse::Ok().body(body),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model = SentenceEmbeddingsBuilder::local("./all-MiniLM-L12-v2").create_model();

    if model.is_err() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Could not load model",
        ));
    }

    let data = web::Data::new(AppState {
        model: Mutex::new(model.unwrap()),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(data.clone())
            .service(sentence_embedding)
    })
    .bind(("localhost", 8080))?
    .run()
    .await
}
