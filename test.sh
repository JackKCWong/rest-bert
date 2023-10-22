#!/usr/bin/env bash

curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["The food was delicious and the waiter", "hello world"],
    "model": "all-MiniLM-L12-v2"
  }'

