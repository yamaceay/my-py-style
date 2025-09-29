.PHONY: build prompt install show-prompt sync-docs

%:
	@:

build:
	@project_name="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$project_name" ]; then \
		echo "Error: project name is required"; \
		echo "Usage: make build <project_name> [type=<type>] [target_dir=<dir>] [author=<author>] [description=<desc>]"; \
		exit 1; \
	fi; \
	echo "Creating project '$$project_name'..."; \
	python3 scaffold.py "$$project_name" \
		$(if $(type),--type $(type)) \
		$(if $(target_dir),--target-dir $(target_dir)) \
		$(if $(author),--author "$(author)") \
		$(if $(description),--description "$(description)") \
		$(if $(no_logging),--no-logging) \
		$(if $(no_cli),--no-cli) \
		$(if $(no_context_managers),--no-context-managers)

install:
	@echo "Installing gh CLI if not already installed..."
	@if ! command -v gh >/dev/null 2>&1; then \
		echo "gh CLI not found. Installing..."; \
		curl -fsSL https://cli.github.com/install.sh | sh; \
	else \
		echo "gh CLI is already installed."; \
	fi
	@echo "Authenticating with GitHub CLI if not already authenticated..."
	@if ! gh auth status >/dev/null 2>&1; then \
		echo "Not authenticated. Running 'gh auth login'..."; \
		gh auth login; \
	else \
		echo "Already authenticated with GitHub CLI."; \
	fi
	@echo "Installing GitHub Copilot CLI extension if not already installed..."
	@if ! gh extension list | grep -q 'github/gh-copilot'; then \
		echo "GitHub Copilot CLI extension not found. Installing..."; \
		gh extension install github/gh-copilot; \
	else \
		echo "GitHub Copilot CLI extension is already installed."; \
	fi

sync-docs:
	@echo "Synchronizing documentation..."
	@python3 scripts/sync-prompt.py
	@echo "Documentation synchronized!"

prompt: show-prompt
	@echo ""
	@echo "🤖 The GitHub Copilot CLI is designed for shell commands, not code transformation."
	@echo "📋 Please copy the prompt above and use it with one of these AI services:"
	@echo ""
	@echo "   • Claude (https://claude.ai)"
	@echo "   • ChatGPT (https://chat.openai.com)"
	@echo "   • GitHub Copilot Chat in VS Code"
	@echo "   • Cursor IDE"
	@echo ""
	@echo "🔗 The prompt has been displayed above - just copy and paste it!"
	@echo "📖 Or view it online: https://yamaceay.github.io/my-py-style/prompt"
	@echo ""
	@read -p "Press Enter to copy prompt to clipboard (macOS)..." && \
	if command -v pbcopy >/dev/null 2>&1; then \
		cat prompt.md | pbcopy && echo "✅ Prompt copied to clipboard!"; \
	else \
		echo "📋 Clipboard copy not available - please copy manually from above."; \
	fi

show-prompt:
	@echo "📋 Go-ish Python Transformation Prompt:"
	@echo "======================================"
	@cat prompt.md
	@echo "======================================"