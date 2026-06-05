#!/bin/bash
#docker compose run --rm chat --context 'db,docs' --context-grep 'md,py,Dockerfile,log,yml,txt' --persist-to-db --experimental-websearch
uv run ./stream_chat.py --context 'db,docs' --context-grep 'md,py,Dockerfile,log,yml,txt' --persist-to-db --experimental-websearch
