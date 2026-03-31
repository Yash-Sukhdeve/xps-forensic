#!/bin/bash
#
# UWS Research Workflow Manager (Self-Contained)
#
# Usage: .uws/scripts/research.sh [action] [details]
#
# Actions:
#   status  - Show current research phase
#   start   - Begin research at hypothesis phase
#   next    - Advance to next phase
#   goto    - Jump to a specific phase
#   reject  - Hypothesis rejected (triggers refinement)
#   reset   - Reset research state
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Research Phase definitions (Scientific Method)
RESEARCH_PHASES=("hypothesis" "literature_review" "experiment_design" "data_collection" "analysis" "peer_review" "publication")
RESEARCH_STATE_KEY="research_phase"

_uws_resolve_workflow_dir || exit 1
STATE_FILE="${WORKFLOW_DIR}/state.yaml"

if [[ ! -f "$STATE_FILE" ]]; then
    echo -e "${UWS_RED}ERROR: ${STATE_FILE} not found${UWS_NC}"
    exit 1
fi

get_current_phase() {
    local phase
    phase=$(_uws_yaml_read "$STATE_FILE" "$RESEARCH_STATE_KEY")
    if [[ -z "$phase" || "$phase" == "null" ]]; then
        echo "not_started"
    else
        echo "$phase"
    fi
}

phase_emoji() {
    case "$1" in
        hypothesis)        echo "💡" ;;
        literature_review) echo "📚" ;;
        experiment_design) echo "🔬" ;;
        data_collection)   echo "📊" ;;
        analysis)          echo "📈" ;;
        peer_review)       echo "👥" ;;
        publication)       echo "📄" ;;
        *)                 echo "❓" ;;
    esac
}

cmd_status() {
    local current
    current=$(get_current_phase)
    echo -e "${UWS_BOLD}${UWS_BLUE}═══════════════════════════════════════${UWS_NC}"
    echo -e "${UWS_BOLD}      Research Workflow Status${UWS_NC}"
    echo -e "${UWS_BOLD}${UWS_BLUE}═══════════════════════════════════════${UWS_NC}"
    echo ""

    if [[ "$current" == "not_started" ]]; then
        echo -e "  Status: ${UWS_YELLOW}Not started${UWS_NC}"
        echo -e "  Run: ${UWS_CYAN}.uws/scripts/research.sh start${UWS_NC}"
        return 0
    fi

    for phase in "${RESEARCH_PHASES[@]}"; do
        local emoji
        emoji=$(phase_emoji "$phase")
        if [[ "$phase" == "$current" ]]; then
            echo -e "  ${emoji} ${UWS_GREEN}${UWS_BOLD}${phase}${UWS_NC} ${UWS_GREEN}<-- CURRENT${UWS_NC}"
        else
            local idx current_idx
            idx=$(_uws_array_index "$phase" "${RESEARCH_PHASES[@]}" || echo "-1")
            current_idx=$(_uws_array_index "$current" "${RESEARCH_PHASES[@]}" || echo "-1")
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
        echo -e "${UWS_YELLOW}Research already started at phase: ${current}${UWS_NC}"
        echo -e "Use ${UWS_CYAN}reset${UWS_NC} first if you want to restart."
        return 1
    fi
    _uws_yaml_write "$STATE_FILE" "$RESEARCH_STATE_KEY" "hypothesis"
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"
    echo -e "${UWS_GREEN}Research started at phase: hypothesis${UWS_NC}"
    echo -e "$(phase_emoji hypothesis) Formulate and document your hypothesis."
}

cmd_next() {
    local current idx next_idx
    current=$(get_current_phase)
    if [[ "$current" == "not_started" ]]; then
        echo -e "${UWS_RED}Research not started. Run: .uws/scripts/research.sh start${UWS_NC}"
        return 1
    fi
    idx=$(_uws_array_index "$current" "${RESEARCH_PHASES[@]}" || true)
    if [[ "$idx" == "-1" ]]; then
        echo -e "${UWS_RED}Unknown current phase: ${current}${UWS_NC}"
        return 1
    fi
    next_idx=$((idx + 1))
    if [[ "$next_idx" -ge "${#RESEARCH_PHASES[@]}" ]]; then
        echo -e "${UWS_GREEN}Research complete! All phases finished.${UWS_NC}"
        echo -e "Current phase remains: ${UWS_BOLD}${current}${UWS_NC}"
        return 0
    fi
    local next_phase="${RESEARCH_PHASES[$next_idx]}"
    _uws_yaml_write "$STATE_FILE" "$RESEARCH_STATE_KEY" "$next_phase"
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"
    echo -e "${UWS_GREEN}Advanced: ${current} -> ${next_phase}${UWS_NC}"
    echo -e "$(phase_emoji "$next_phase") Now in ${UWS_BOLD}${next_phase}${UWS_NC} phase."
}

cmd_goto() {
    local target="${1:-}"
    if [[ -z "$target" ]]; then
        echo -e "${UWS_RED}Usage: .uws/scripts/research.sh goto <phase>${UWS_NC}"
        echo -e "Phases: ${RESEARCH_PHASES[*]}"
        return 1
    fi
    local idx
    idx=$(_uws_array_index "$target" "${RESEARCH_PHASES[@]}" || true)
    if [[ "$idx" == "-1" ]]; then
        echo -e "${UWS_RED}Unknown phase: ${target}${UWS_NC}"
        echo -e "Valid phases: ${RESEARCH_PHASES[*]}"
        return 1
    fi
    local current
    current=$(get_current_phase)
    _uws_yaml_write "$STATE_FILE" "$RESEARCH_STATE_KEY" "$target"
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"
    echo -e "${UWS_GREEN}Jumped: ${current} -> ${target}${UWS_NC}"
}

cmd_reject() {
    local reason="${1:-hypothesis not supported by evidence}"
    local current
    current=$(get_current_phase)
    if [[ "$current" == "not_started" ]]; then
        echo -e "${UWS_RED}Research not started.${UWS_NC}"
        return 1
    fi

    # Rejection sends back to hypothesis for refinement
    _uws_yaml_write "$STATE_FILE" "$RESEARCH_STATE_KEY" "hypothesis"
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"

    local ts
    ts=$(_uws_timestamp)
    echo "${ts} | RESEARCH_REJECT | ${current} -> hypothesis | ${reason}" >> "${WORKFLOW_DIR}/checkpoints.log"

    echo -e "${UWS_RED}Rejected at ${current}: ${reason}${UWS_NC}"
    echo -e "${UWS_YELLOW}Returned to hypothesis phase for refinement.${UWS_NC}"
}

cmd_reset() {
    if grep -q "^${RESEARCH_STATE_KEY}:" "$STATE_FILE" 2>/dev/null; then
        sed -i.bak "/^${RESEARCH_STATE_KEY}:/d" "$STATE_FILE"
        rm -f "${STATE_FILE}.bak"
    fi
    _uws_yaml_write "$STATE_FILE" "last_updated" "$(_uws_timestamp)"
    echo -e "${UWS_GREEN}Research state reset.${UWS_NC}"
}

ACTION="${1:-status}"
shift || true

case "$ACTION" in
    status) cmd_status ;;
    start)  cmd_start ;;
    next)   cmd_next ;;
    goto)   cmd_goto "$@" ;;
    reject) cmd_reject "$*" ;;
    reset)  cmd_reset ;;
    *)
        echo -e "${UWS_RED}Unknown action: ${ACTION}${UWS_NC}"
        echo "Usage: research.sh {status|start|next|goto|reject|reset}"
        exit 1
        ;;
esac
