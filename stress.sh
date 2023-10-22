#!/usr/bin/env bash


dur=${1:-5s}

cat req.http | vegeta attack -duration="$dur" | vegeta report

