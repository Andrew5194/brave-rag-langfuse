# Brave RAG + Langfuse — workspace tasks.
#
#   make local   Start the stack and Jupyter for local use (localhost access).
#   make remote  Start the stack, open the Coder port forwarder, and bind
#                Jupyter on 0.0.0.0 so it is reachable through the Coder proxy.
#   make down    Stop the stack.
#
# "remote" is for this Coder workspace, where the Langfuse services run as
# sibling containers: coder-connect.sh attaches this workspace to their network
# and starts socat forwarders so localhost:3005/9090 reach them. "local" skips
# that — use it where Docker already publishes the ports to your localhost.

NETWORK := brave-rag-langfuse_default

.PHONY: local remote down

local:
	docker compose up -d
	uv run jupyter-lab

remote:
	docker compose up -d
	./coder-connect.sh
	uv run jupyter-lab --ip 0.0.0.0

down:
	-pkill -x socat
	-docker network disconnect $(NETWORK) "$$(grep '/containers/' /proc/self/mountinfo | grep -oE '[0-9a-f]{64}' | head -1)"
	docker compose down
