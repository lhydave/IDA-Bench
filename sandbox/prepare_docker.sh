#!/bin/bash

# Create necessary directories
mkdir -p configs instructions llms

# Copy files
cp ../llm_config_agent.toml agent_config.toml
cp ../configs/base_config.toml configs/
cp ../configs/interpreter_config.toml configs/
cp ../instructions/gatekeeper_reference.md instructions/
cp ../logger.py .
cp ../llm_interact_env.py .
cp ../llms/llm_interact.py llms/

echo "Files copied successfully!" 