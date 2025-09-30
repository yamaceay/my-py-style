.PHONY: build prompt show-prompt sync-docs

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
	@echo "📖 Or view it online: https://yamaceay.github.io/py-kit/prompt"
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