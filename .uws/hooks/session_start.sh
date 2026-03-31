#!/bin/bash
# UWS SessionStart Hook - Silently inject workflow context into Claude

WORKFLOW_DIR="${CLAUDE_PROJECT_DIR:-.}/.workflow"

# Exit silently if no workflow
[[ ! -d "$WORKFLOW_DIR" ]] && exit 0

# Build context from state files
CONTEXT=""

# Read state.yaml
if [[ -f "$WORKFLOW_DIR/state.yaml" ]]; then
    PHASE=$(grep -E "^current_phase:" "$WORKFLOW_DIR/state.yaml" 2>/dev/null | cut -d: -f2 | tr -d ' "' || echo "unknown")
    CHECKPOINT=$(grep -E "^current_checkpoint:" "$WORKFLOW_DIR/state.yaml" 2>/dev/null | cut -d: -f2 | tr -d ' "' || echo "none")
    PROJECT_TYPE=$(grep -E "^  type:" "$WORKFLOW_DIR/state.yaml" 2>/dev/null | head -1 | cut -d: -f2 | tr -d ' "' || echo "unknown")

    CONTEXT+="## Workflow State\n"
    CONTEXT+="- Phase: ${PHASE}\n"
    CONTEXT+="- Checkpoint: ${CHECKPOINT}\n"
    CONTEXT+="- Project Type: ${PROJECT_TYPE}\n\n"
fi

# Read recent checkpoints
if [[ -f "$WORKFLOW_DIR/checkpoints.log" ]]; then
    RECENT=$(tail -3 "$WORKFLOW_DIR/checkpoints.log" 2>/dev/null | grep -v "^#" || echo "")
    if [[ -n "$RECENT" ]]; then
        CONTEXT+="## Recent Checkpoints\n\`\`\`\n${RECENT}\n\`\`\`\n\n"
    fi
fi

# Read priority actions from handoff
if [[ -f "$WORKFLOW_DIR/handoff.md" ]]; then
    ACTIONS=$(sed -n '/^## Next Actions/,/^##/p' "$WORKFLOW_DIR/handoff.md" 2>/dev/null | head -10 | grep -E "^-|\[" || echo "")
    if [[ -n "$ACTIONS" ]]; then
        CONTEXT+="## Priority Actions\n${ACTIONS}\n\n"
    fi
fi

# Git status (only if git is available and this is a repo)
if command -v git &>/dev/null && git rev-parse --git-dir &>/dev/null 2>&1; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "")
    MODIFIED=$(git status --porcelain 2>/dev/null | grep -c "^ M" || echo "0")
    if [[ -n "$BRANCH" ]]; then
        CONTEXT+="## Git\n- Branch: ${BRANCH}\n- Modified files: ${MODIFIED}\n\n"
    fi
fi

# Output as JSON for Claude to consume
if [[ -n "$CONTEXT" ]]; then
    CONTEXT_ESCAPED=$(echo -e "$CONTEXT" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    echo "{\"additionalContext\": \"${CONTEXT_ESCAPED}\"}"
fi

exit 0
