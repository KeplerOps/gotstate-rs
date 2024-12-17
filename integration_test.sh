#!/usr/bin/env bash

python -m pytest hsm/tests/test_integration.py -v --log-cli-level=DEBUG
