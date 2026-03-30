#!/bin/bash
set -e

MODEL="${1:-models/Qwen3-8B-Q4_K_M.gguf}"
WIKI="${2:-wikitext-2-raw/wiki.test.raw}"
CHUNKS="${3:-8}"
CTX="${4:-512}"

echo "Model: $MODEL"
echo "Dataset: $WIKI"
echo "Chunks: $CHUNKS, Context: $CTX"
echo ""

echo "========== 5hi_3lo_had PPL =========="
for V in tqv_had_mse4 q4_0; do
    echo -n "  tqk_5hi_3lo_had + $V: "
    ./build/bin/llama-perplexity -m "$MODEL" -f "$WIKI" -ctk tqk_5hi_3lo_had -ctv "$V" -ngl 99 --flash-attn on --chunks $CHUNKS -c $CTX 2>&1 | grep "Final"
done

echo ""
echo "========== 5hi_3lo_had calibration windows =========="
for V in tqv_had_mse4 q4_0; do
    echo "--- V = $V ---"
    for SINK in 32 64 128 256 512; do
        echo -n "  sink=$SINK: "
        ./build/bin/llama-perplexity -m "$MODEL" -f "$WIKI" -ctk tqk_5hi_3lo_had -ctv "$V" --tq-sinks $SINK -ngl 99 --flash-attn on --chunks $CHUNKS -c $CTX 2>&1 | grep "Final\|accumulated" | tr '\n' ' '
        echo ""
    done
done

echo ""
echo "========== Sanity check (had_mse4 + tqv — not affected by RoPE) =========="
echo -n "  tqk_had_mse4 + tqv_had_mse4: "
./build/bin/llama-perplexity -m "$MODEL" -f "$WIKI" -ctk tqk_had_mse4 -ctv tqv_had_mse4 -ngl 99 --flash-attn on --chunks $CHUNKS -c $CTX 2>&1 | grep "Final"
