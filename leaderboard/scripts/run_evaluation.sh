#!/bin/bash

export LOCAL_ROUTES=/workspace/leaderboard/data/debug.xml
export LOCAL_SCENARIOS=/workspace/leaderboard/data/scenarios/eval_scenarios.json

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${LOCAL_SCENARIOS}  \
--routes=${LOCAL_ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME}

