-include .env

export USER_ID
export GROUP_ID
export USERNAME
export ROS_DOMAIN_ID
export SERVER_IP
export ORIN_IP
export VERSION
export DOCKER_BUILDKIT=1

PROJECT_NAME ?= gst_realsense

SENDER_SCRIPT := src/startup/sender.py
RECEIVER_SCRIPT := src/startup/receiver.py
CONFIG_FILE := src/config/config.yaml
DOCKER_COMPOSE ?= docker compose

.PHONY: build build-orin build-server
build: build-orin build-server

build-orin:
	$(DOCKER_COMPOSE) --profile orin build

build-server:
	$(DOCKER_COMPOSE) --profile server build

.PHONY: up-orin up-server
up-orin:
	$(DOCKER_COMPOSE) --profile orin up -d

up-server:
	$(DOCKER_COMPOSE) --profile server up -d

.PHONY: run-send-all
run-send-all:
	$(DOCKER_COMPOSE) --profile orin run --rm orin \
		python3 $(SENDER_SCRIPT) --all --config $(CONFIG_FILE)

.PHONY: run-receive-all
run-receive-all:
	$(DOCKER_COMPOSE) --profile server run --rm server \
		python3 $(RECEIVER_SCRIPT) --all --config $(CONFIG_FILE)

.PHONY: run-diagnostic
run-diagnostic:
	@echo "---Run the network diagnostic tool ---"
	@echo "Note: Please ensure that the sending end is running at $(ORIN_IP) and the receiving end IP is $(SERVER_IP)"
	$(DOCKER_COMPOSE) --profile orin run --rm orin \
		python3 src/checker/diagnose_stream.py

.PHONY: clean
clean:
	@echo "--- clean Docker service ---"
	$(DOCKER_COMPOSE) down --rmi all
	@echo "--- clean finished ---"

.PHONY: info
info:
	@echo "=== Version Information ==="
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "=== System Information ==="
	@echo "User: $(USERNAME) ($(USER_ID):$(GROUP_ID))"
	@echo ""
	@echo "=== Network Configuration ==="
	@echo "ROS Domain ID: $(ROS_DOMAIN_ID)"
	@echo "Server IP: $(SERVER_IP)"
	@echo "Orin IP: $(ORIN_IP)"
	@echo ""
	@echo "=== Docker Information ==="
	@docker version || echo "Docker not available"
	@echo ""
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "=== NVIDIA GPU Information ==="; \
		nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; \
	else \
		echo "No NVIDIA GPU detected"; \
	fi
