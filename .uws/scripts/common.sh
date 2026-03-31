#!/bin/bash
# UWS Common Utilities - Shared by all workflow scripts

# Colors
readonly UWS_GREEN='\033[0;32m'
readonly UWS_YELLOW='\033[1;33m'
readonly UWS_RED='\033[0;31m'
readonly UWS_CYAN='\033[0;36m'
readonly UWS_BLUE='\033[0;34m'
readonly UWS_BOLD='\033[1m'
readonly UWS_NC='\033[0m'

# Resolve workflow directory
_uws_resolve_workflow_dir() {
    if [[ -n "${WORKFLOW_DIR:-}" ]]; then
        return 0
    fi
    if [[ -d "$(pwd)/.workflow" ]]; then
        WORKFLOW_DIR="$(pwd)/.workflow"
        return 0
    fi
    if command -v git &>/dev/null; then
        local git_root
        git_root="$(git rev-parse --show-toplevel 2>/dev/null)" || true
        if [[ -n "$git_root" && -d "${git_root}/.workflow" ]]; then
            WORKFLOW_DIR="${git_root}/.workflow"
            return 0
        fi
    fi
    echo -e "${UWS_RED}ERROR: No .workflow/ directory found.${UWS_NC}" >&2
    echo -e "Run the UWS installer first or cd to your project root." >&2
    return 1
}

# Read a YAML value (simple grep-based, no yq dependency)
_uws_yaml_read() {
    local file="$1" key="$2"
    grep -E "^${key}:" "$file" 2>/dev/null | head -1 | cut -d: -f2- | tr -d ' "' || echo ""
}

# Write a YAML value (simple sed-based)
_uws_yaml_write() {
    local file="$1" key="$2" value="$3"
    if grep -qE "^${key}:" "$file" 2>/dev/null; then
        sed -i.bak "s|^${key}:.*|${key}: \"${value}\"|" "$file"
        rm -f "${file}.bak"
    else
        echo "${key}: \"${value}\"" >> "$file"
    fi
}

# Get current timestamp
_uws_timestamp() {
    date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S
}

# Find array index (-1 if not found)
_uws_array_index() {
    local needle="$1"; shift
    local arr=("$@")
    for i in "${!arr[@]}"; do
        if [[ "${arr[$i]}" == "$needle" ]]; then
            echo "$i"
            return 0
        fi
    done
    echo "-1"
    return 1
}
