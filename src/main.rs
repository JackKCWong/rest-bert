mod model;
mod app;

use std::{
    env,
    sync::{Arc, Mutex},
};

use actix_web::{
    post,
    web::{self},
    App, HttpResponse, HttpServer, Responder,
};
use async_object_pool::Pool;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder,
};
use serde::{Deserialize, Serialize};


use app::*;
use model::*;

#[post("/v1/embeddings")]
async fn sentence_embedding(
    app: web::Data<AppState>,
    form: web::Json<EmbeddingRequest>,
) -> impl Responder {
    let model = app
        .models
        .take_or_create(|| {
            Arc::new(Mutex::new(
                SentenceEmbeddingsBuilder::local("./all-MiniLM-L12-v2")
                    .create_model()
                    .unwrap(),
            ))
        })
        .await;

    let model_rc = model.clone();
    let tensors = web::block(move || model.lock().unwrap().encode(&form.inputs).unwrap()).await;

    app.models.put(model_rc).await;

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

    let pool = Pool::new(4);

    let data = web::Data::new(AppState { models: pool });

    HttpServer::new(move || {
        App::new()
            .app_data(data.clone())
            .service(sentence_embedding)
    })
    .bind(("localhost", 8080))?
    .run()
    .await
}
