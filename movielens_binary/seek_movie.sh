#! /bin/sh


while read id; do grep "^$id," data/ml-20m/movies.csv ; done;
