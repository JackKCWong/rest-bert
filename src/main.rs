use std::{
    env,
    sync::Mutex,
};

use actix_web::{
    post,
    web::{self},
    App, HttpResponse, HttpServer, Responder,
};
use pool::{Dirty, Pool};
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
    object: &'static str,
    data: Vec<Embedding>,
    model: &'static str,
}

#[derive(Deserialize, Serialize)]
struct Embedding {
    object: &'static str,
    embedding: Vec<f32>,
    index: usize,
}

struct AppState {
    models: Mutex<Pool<Dirty<SentenceEmbeddingsModel>>>,
}

#[post("/v1/embeddings")]
async fn sentence_embedding(
    app: web::Data<AppState>,
    form: web::Json<EmbeddingRequest>,
) -> impl Responder {
    let model = app.models.lock().unwrap().checkout().unwrap();
    let model = Mutex::new(model);

    let tensors = web::block(move || model.lock().unwrap().encode(&form.input).unwrap()).await;
    if tensors.is_err() {
        return HttpResponse::InternalServerError().body("Could not embed input");
    }

    let tensors = tensors.unwrap();
    let mut embeddings = Vec::with_capacity(tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        embeddings.push(Embedding {
            object: "embedding",
            embedding: t.to_vec(),
            index: i,
        });
    }

    match serde_json::to_string(&EmbeddingResponse {
        model: "all-MiniLM-L12-v2",
        object: "list",
        data: embeddings,
    }) {
        Ok(body) => HttpResponse::Ok().body(body),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    for (key, value) in env::vars() {
        println!("{key}: {value}");
    }

    // let model = SentenceEmbeddingsBuilder::local("./all-MiniLM-L12-v2").create_model();

    // if model.is_err() {
    //     return Err(std::io::Error::new(
    //         std::io::ErrorKind::Other,
    //         "Could not load model",
    //     ));
    // }

    let pool = Pool::with_capacity(6, 0, || {
        Dirty(
            SentenceEmbeddingsBuilder::local("./all-MiniLM-L12-v2")
                .create_model()
                .unwrap(),
        )
    });

    let data = web::Data::new(AppState {
        models: Mutex::new(pool),
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
