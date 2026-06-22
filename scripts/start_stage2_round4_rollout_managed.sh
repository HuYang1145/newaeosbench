#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AEOS_ENV_BIN="${AEOS_ENV_BIN:-/home/hy/miniconda3/envs/aeos/bin}"
RUN_NAME="${RUN_NAME:-stage4_from_old_200k}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
CANDIDATE_EPOCH="${CANDIDATE_EPOCH:-4}"
TAU_E="${TAU_E:-4.5}"
MODEL_DEVICE="${MODEL_DEVICE:-0}"
ROLLOUT_DEVICE="${ROLLOUT_DEVICE:-cpu}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-}"
ROLLOUT_NICE="${ROLLOUT_NICE:-19}"
UNIT_NAME="${UNIT_NAME:-stage4-rollout-${RUN_NAME}}"
SESSION_NAME="${SESSION_NAME:-${UNIT_NAME}}"
LAUNCHER="${LAUNCHER:-auto}"

LOG_DIR="work_dirs/${RUN_NAME}/managed"
LAUNCH_LOG="${LOG_DIR}/${UNIT_NAME}.log"
PID_FILE="${LOG_DIR}/${UNIT_NAME}.pid"
CMD_FILE="${LOG_DIR}/${UNIT_NAME}.command.sh"

mkdir -p "${LOG_DIR}"

RUN_COMMAND="cd '${ROOT_DIR}' && env AEOS_ENV_BIN='${AEOS_ENV_BIN}' RUN_NAME='${RUN_NAME}' TRAIN_SPLIT='${TRAIN_SPLIT}' CANDIDATE_EPOCH='${CANDIDATE_EPOCH}' TAU_E='${TAU_E}' MODEL_DEVICE='${MODEL_DEVICE}' ROLLOUT_DEVICE='${ROLLOUT_DEVICE}' ROLLOUT_LIMIT='${ROLLOUT_LIMIT}' ROLLOUT_WORKERS='${ROLLOUT_WORKERS}' ROLLOUT_NICE='${ROLLOUT_NICE}' STOP_AFTER_ROLLOUT=1 bash scripts/start_stage2_round4_parallel.sh"

cat > "${CMD_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
${RUN_COMMAND}
EOF
chmod +x "${CMD_FILE}"

pick_launcher() {
  if [[ "${LAUNCHER}" != "auto" ]]; then
    echo "${LAUNCHER}"
    return
  fi

  local linger
  linger="$(loginctl show-user "$(id -un)" --property=Linger --value 2>/dev/null || true)"
  if [[ "${linger}" == "yes" ]] && command -v systemd-run >/dev/null 2>&1; then
    echo "systemd"
    return
  fi

  if command -v tmux >/dev/null 2>&1; then
    echo "tmux"
    return
  fi

  echo "nohup"
}

is_running() {
  local pid
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

if is_running; then
  echo "[error] managed rollout already running with pid $(cat "${PID_FILE}")" >&2
  echo "[hint] file log: ${LAUNCH_LOG}" >&2
  exit 1
fi

SELECTED_LAUNCHER="$(pick_launcher)"

echo "[info] launcher: ${SELECTED_LAUNCHER}"
echo "[info] unit/session: ${UNIT_NAME}"
echo "[info] launch log: ${LAUNCH_LOG}"
echo "[info] pid file: ${PID_FILE}"
echo "[info] command file: ${CMD_FILE}"
echo "[info] candidate trajectory root: data/trajectories.${CANDIDATE_EPOCH}"
echo "[info] rollout resumes safely by skipping existing trajectories"

case "${SELECTED_LAUNCHER}" in
  systemd)
    if systemctl --user --quiet is-active "${UNIT_NAME}.service"; then
      echo "[error] managed rollout already running: ${UNIT_NAME}.service" >&2
      exit 1
    fi
    systemd-run --user \
      --unit="${UNIT_NAME}" \
      --same-dir \
      --collect \
      --property=WorkingDirectory="${ROOT_DIR}" \
      --property=KillMode=control-group \
      /bin/bash -lc "${RUN_COMMAND} >> '${LAUNCH_LOG}' 2>&1"
    echo "[done] managed rollout submitted with systemd"
    echo "[done] status: systemctl --user status ${UNIT_NAME}.service"
    echo "[done] logs: journalctl --user -u ${UNIT_NAME}.service -n 200"
    echo "[done] file log: ${LAUNCH_LOG}"
    ;;
  tmux)
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
      echo "[error] tmux session already exists: ${SESSION_NAME}" >&2
      echo "[hint] attach with: tmux attach -t ${SESSION_NAME}" >&2
      exit 1
    fi
    tmux new-session -d -s "${SESSION_NAME}" \
      "cd '${ROOT_DIR}' && bash '${CMD_FILE}' >> '${LAUNCH_LOG}' 2>&1"
    tmux list-panes -t "${SESSION_NAME}" -F '#{pane_pid}' > "${PID_FILE}"
    echo "[done] managed rollout submitted with tmux"
    echo "[done] session: ${SESSION_NAME}"
    echo "[done] attach: tmux attach -t ${SESSION_NAME}"
    echo "[done] file log: ${LAUNCH_LOG}"
    ;;
  nohup)
    nohup /bin/bash -lc "cd '${ROOT_DIR}' && bash '${CMD_FILE}' >> '${LAUNCH_LOG}' 2>&1" >/dev/null 2>&1 &
    echo "$!" > "${PID_FILE}"
    echo "[done] managed rollout submitted with nohup"
    echo "[done] pid: $(cat "${PID_FILE}")"
    echo "[done] file log: ${LAUNCH_LOG}"
    ;;
  *)
    echo "[error] unsupported launcher: ${SELECTED_LAUNCHER}" >&2
    exit 1
    ;;
esac
