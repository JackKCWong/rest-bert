use actix_web::{post, web, App, HttpResponse, HttpServer, Responder};
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;
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

#[post("/v1/embeddings")]
async fn sentence_embedding(form: web::Json<EmbeddingRequest>) -> impl Responder {
    let model = SentenceEmbeddingsBuilder::local("./all-MiniLM-L12-v2")
        .create_model();

    if model.is_err() {
        return HttpResponse::InternalServerError().body("Could not load model");
    }

    let model = model.unwrap();
    let embeddings = model.encode(&form.input);
    if embeddings.is_err() {
        return HttpResponse::InternalServerError().body("Could not embed input");
    }

    let output = embeddings.unwrap();
    let mut embeddings = Vec::new();
    for (i, e) in output.iter().enumerate() {
        embeddings.push(Embedding {
            object: "embedding".to_owned(),
            embedding: e.to_vec(),
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
    HttpServer::new(|| App::new().service(sentence_embedding))
        .bind(("localhost", 8080))?
        .run()
        .await
}
