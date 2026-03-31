#!/bin/bash
# UWS PreCompact Hook - Auto-checkpoint before context compaction

WORKFLOW_DIR="${CLAUDE_PROJECT_DIR:-.}/.workflow"
CHECKPOINT_LOG="$WORKFLOW_DIR/checkpoints.log"

# Exit silently if no workflow
[[ ! -d "$WORKFLOW_DIR" ]] && exit 0

# Get current checkpoint number
if [[ -f "$CHECKPOINT_LOG" ]]; then
    LAST_CP=$(grep -oE "CP_[0-9]+_[0-9]+" "$CHECKPOINT_LOG" | tail -1 || echo "CP_1_000")
    PHASE=$(echo "$LAST_CP" | cut -d_ -f2)
    SEQ=$(echo "$LAST_CP" | cut -d_ -f3 | sed 's/^0*//')
    NEW_SEQ=$(printf "%03d" $((SEQ + 1)))
    NEW_CP="CP_${PHASE}_${NEW_SEQ}"
else
    NEW_CP="CP_1_001"
fi

# Create auto-checkpoint
TIMESTAMP=$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)
echo "${TIMESTAMP} | ${NEW_CP} | Auto-checkpoint before context compaction" >> "$CHECKPOINT_LOG"

# Update state.yaml checkpoint
if [[ -f "$WORKFLOW_DIR/state.yaml" ]]; then
    sed -i.bak "s/current_checkpoint:.*/current_checkpoint: \"${NEW_CP}\"/" "$WORKFLOW_DIR/state.yaml" 2>/dev/null || true
    sed -i.bak "s/last_updated:.*/last_updated: \"${TIMESTAMP}\"/" "$WORKFLOW_DIR/state.yaml" 2>/dev/null || true
    rm -f "$WORKFLOW_DIR/state.yaml.bak"
fi

echo "{\"status\": \"checkpoint_created\", \"checkpoint\": \"${NEW_CP}\"}"
exit 0
