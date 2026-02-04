#!/bin/bash
docker compose run --rm chat --context . --context-grep 'md,py,Dockerfile,log,yml' --persist-to-db --select-session
