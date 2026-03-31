#!/bin/bash
#
# UWS SDLC Workflow Manager (Self-Contained)
#
# Usage: .uws/scripts/sdlc.sh [action] [details]
#
# Actions:
#   status  - Show current SDLC phase
#   start   - Begin SDLC at requirements phase
#   next    - Advance to next phase
#   goto    - Jump to a specific phase
#   fail    - Report failure (triggers regression to previous phase)
#   reset   - Reset SDLC state
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# SDLC Phase definitions
SDLC_PHASES=("requirements" "design" "implementation" "verification" "deployment" "maintenance")
SDLC_STATE_KEY="sdlc_phase"

_uws_resolve_workflow_dir || exit 1
STATE_FILE="${WORKFLOW_DIR}/state.yaml"

# Ensure state file exists
if [[ ! -f "$STATE_FILE" ]]; then
    echo -e "${UWS_RED}ERROR: ${STATE_FILE} not found${UWS_NC}"
    exit 1
fi

# Get current SDLC phase from state
get_current_phase() {
    local phase
    phase=$(_uws_yaml_read "$STATE_FILE" "$SDLC_STATE_KEY")
    if [[ -z "$phase" || "$phase" == "null" ]]; then
        echo "not_started"
    else
        echo "$phase"
    fi
}

# Phase emoji
phase_emoji() {
    case "$1" in
        requirements)   echo "📋" ;;
        design)         echo "🏗️" ;;
        implementation) echo "💻" ;;
        verification)   echo "✅" ;;
        deployment)     echo "🚀" ;;
        maintenance)    echo "🔧" ;;
        *)              echo "❓" ;;
    esac
}

cmd_status() {
    local current
    current=$(get_current_phase)
    echo -e "${UWS_BOLD}${UWS_BLUE}═══════════════════════════════════════${UWS_NC}"
    echo -e "${UWS_BOLD}        SDLC Workflow Status${UWS_NC}"
    echo -e "${UWS_BOLD}${UWS_BLUE}═══════════════════════════════════════${UWS_NC}"
    echo ""

    if [[ "$current" == "not_started" ]]; then
        echo -e "  Status: ${UWS_YELLOW}Not started${UWS_NC}"
        echo -e "  Run: ${UWS_CYAN}.uws/scripts/sdlc.sh start${UWS_NC}"
        return 0
    fi

    for phase in "${SDLC_PHASES[@]}"; do
        local emoji
        emoji=$(phase_emoji "$phase")
        if [[ "$phase" == "$current" ]]; then
            echo -e "  ${emoji} ${UWS_GREEN}${UWS_BOLD}${phase}${UWS_NC} ${UWS_GREEN}<-- CURRENT${UWS_NC}"
        else
            local idx current_idx
            idx=$(_uws_array_index "$phase" "${SDLC_PHASES[@]}" || echo "-1")
            current_idx=$(_uws_array_index "$current" "${SDLC_PHASES[@]}" || echo "-1")
            if [[ "$idx" -lt "$current_idx" ]]; then
                echo -e "  ${emoji} ${phase} ${UWS_CYAN}(done)${UWS_NC}"
            else
                echo -e "  ${emoji} ${phase}"
            fi
        fi
    done
    echo ""
}

cmd_start() {
    local current
    current=$(get_current_phase)
    if [[ "$current" != "not_started" ]]; then
        echo -e "${UWS_YELLOW}SDLC already started at phase: ${current}${UWS_NC}"
        echo -e "Use ${UWS_CYAN}reset${UWS_NC} first if you want to restart."
        return 1
    fi
    _uws_yaml_write "$STATE_FILE" "$SDLC_STATE_KEY" "requirements"
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"
    echo -e "${UWS_GREEN}SDLC started at phase: requirements${UWS_NC}"
    echo -e "$(phase_emoji requirements) Gather and document requirements."
}

cmd_next() {
    local current idx next_idx
    current=$(get_current_phase)
    if [[ "$current" == "not_started" ]]; then
        echo -e "${UWS_RED}SDLC not started. Run: .uws/scripts/sdlc.sh start${UWS_NC}"
        return 1
    fi
    idx=$(_uws_array_index "$current" "${SDLC_PHASES[@]}" || true)
    if [[ "$idx" == "-1" ]]; then
        echo -e "${UWS_RED}Unknown current phase: ${current}${UWS_NC}"
        return 1
    fi
    next_idx=$((idx + 1))
    if [[ "$next_idx" -ge "${#SDLC_PHASES[@]}" ]]; then
        echo -e "${UWS_GREEN}SDLC complete! All phases finished.${UWS_NC}"
        echo -e "Current phase remains: ${UWS_BOLD}${current}${UWS_NC}"
        return 0
    fi
    local next_phase="${SDLC_PHASES[$next_idx]}"
    _uws_yaml_write "$STATE_FILE" "$SDLC_STATE_KEY" "$next_phase"
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"
    echo -e "${UWS_GREEN}Advanced: ${current} -> ${next_phase}${UWS_NC}"
    echo -e "$(phase_emoji "$next_phase") Now in ${UWS_BOLD}${next_phase}${UWS_NC} phase."
}

cmd_goto() {
    local target="${1:-}"
    if [[ -z "$target" ]]; then
        echo -e "${UWS_RED}Usage: .uws/scripts/sdlc.sh goto <phase>${UWS_NC}"
        echo -e "Phases: ${SDLC_PHASES[*]}"
        return 1
    fi
    local idx
    idx=$(_uws_array_index "$target" "${SDLC_PHASES[@]}" || true)
    if [[ "$idx" == "-1" ]]; then
        echo -e "${UWS_RED}Unknown phase: ${target}${UWS_NC}"
        echo -e "Valid phases: ${SDLC_PHASES[*]}"
        return 1
    fi
    local current
    current=$(get_current_phase)
    _uws_yaml_write "$STATE_FILE" "$SDLC_STATE_KEY" "$target"
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"
    echo -e "${UWS_GREEN}Jumped: ${current} -> ${target}${UWS_NC}"
}

cmd_fail() {
    local reason="${1:-unspecified}"
    local current idx
    current=$(get_current_phase)
    if [[ "$current" == "not_started" ]]; then
        echo -e "${UWS_RED}SDLC not started.${UWS_NC}"
        return 1
    fi
    idx=$(_uws_array_index "$current" "${SDLC_PHASES[@]}" || true)
    if [[ "$idx" -le 0 ]]; then
        echo -e "${UWS_YELLOW}Already at first phase. Cannot regress further.${UWS_NC}"
        echo -e "Failure noted: ${reason}"
        return 0
    fi
    local prev_phase="${SDLC_PHASES[$((idx - 1))]}"
    _uws_yaml_write "$STATE_FILE" "$SDLC_STATE_KEY" "$prev_phase"
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"

    # Log failure
    local ts
    ts=$(_uws_timestamp)
    echo "${ts} | SDLC_FAIL | ${current} -> ${prev_phase} | ${reason}" >> "${WORKFLOW_DIR}/checkpoints.log"

    echo -e "${UWS_RED}Failure in ${current}: ${reason}${UWS_NC}"
    echo -e "${UWS_YELLOW}Regressed to: ${prev_phase}${UWS_NC}"
}

cmd_reset() {
    # Remove SDLC phase from state
    if grep -q "^${SDLC_STATE_KEY}:" "$STATE_FILE" 2>/dev/null; then
        sed -i.bak "/^${SDLC_STATE_KEY}:/d" "$STATE_FILE"
        rm -f "${STATE_FILE}.bak"
    fi
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"
    echo -e "${UWS_GREEN}SDLC state reset.${UWS_NC}"
}

# Main dispatch
ACTION="${1:-status}"
shift || true

case "$ACTION" in
    status) cmd_status ;;
    start)  cmd_start ;;
    next)   cmd_next ;;
    goto)   cmd_goto "$@" ;;
    fail)   cmd_fail "$*" ;;
    reset)  cmd_reset ;;
    *)
        echo -e "${UWS_RED}Unknown action: ${ACTION}${UWS_NC}"
        echo "Usage: sdlc.sh {status|start|next|goto|fail|reset}"
        exit 1
        ;;
esac
