#!/bin/bash
docker compose run --rm chat --dest sessions/session-$(datestr).log --context . --context-grep 'md,py,Dockerfile,log,yml' --persist-to-db
