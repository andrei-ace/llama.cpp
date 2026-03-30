#!/bin/bash
set -e

MODEL="${1:-models/Qwen3-8B-Q4_K_M.gguf}"
WIKI="${2:-wikitext-2-raw/wiki.test.raw}"
CHUNKS="${3:-8}"
CTX="${4:-512}"
OUTDIR="${5:-article}"

# Extract model name for CSV filenames
MNAME=$(basename "$MODEL" | sed 's/-Q4_K_M\.gguf//; s/-instruct//; s/-q4_k_m.*//; s/Meta-//')

echo "Model: $MODEL ($MNAME)"
echo "Dataset: $WIKI"
echo "Chunks: $CHUNKS, Context: $CTX"
echo "Output: $OUTDIR/"
echo ""

COMBOS=(
    # baselines
    "f16:f16"
    "q8_0:q8_0"
    "q4_0:q4_0"
    # 5hi_3lo_had (3.88 bpv K)
    "tqk_5hi_3lo_had:tqv_had_mse4"
    "tqk_5hi_3lo_had:q4_0"
    # had_mse4 (4.13 bpv K)
    "tqk_had_mse4:tqv_had_mse4"
    "tqk_had_mse4:q4_0"
    # had_prod4 (4.25 bpv K)
    "tqk_had_prod4:tqv_had_mse4"
    "tqk_had_prod4:q4_0"
    # had_prod5 (5.25 bpv K)
    "tqk_had_prod5:tqv_had_mse4"
    "tqk_had_prod5:q4_0"
)

# bpv lookup
bpv_k() {
    case "$1" in
        f16) echo "16.00";; q8_0) echo "8.00";; q4_0) echo "4.50";;
        tqk_5hi_3lo_had) echo "3.88";; tqk_had_mse4) echo "4.13";;
        tqk_had_prod4) echo "4.25";; tqk_had_prod5) echo "5.25";;
    esac
}
bpv_v() {
    case "$1" in
        f16) echo "16.00";; q8_0) echo "8.00";; q4_0) echo "4.50";;
        tqv_had_mse4) echo "4.13";;
    esac
}

# --- PPL ---
CSV="$OUTDIR/benchmark-${MNAME}.csv"
echo "model,k_type,v_type,k_bpv,v_bpv,ppl,pp512,tg128" > "$CSV"

echo "========== PPL (${CHUNKS} chunks, c=${CTX}) =========="
for COMBO in "${COMBOS[@]}"; do
    K=$(echo $COMBO | cut -d: -f1)
    V=$(echo $COMBO | cut -d: -f2)
    echo -n "  $K + $V: "
    PPL=$(./build/bin/llama-perplexity -m "$MODEL" -f "$WIKI" -ctk "$K" -ctv "$V" -ngl 99 --flash-attn on --chunks $CHUNKS -c $CTX 2>&1 | grep "Final" | sed 's/.*PPL = \([0-9.]*\).*/\1/')
    echo "PPL = $PPL"
    echo "$MNAME,$K,$V,$(bpv_k $K),$(bpv_v $V),$PPL,," >> "$CSV"
done

# --- Speed ---
echo ""
echo "========== SPEED (pp512 + tg128) =========="
for COMBO in "${COMBOS[@]}"; do
    K=$(echo $COMBO | cut -d: -f1)
    V=$(echo $COMBO | cut -d: -f2)
    BENCH=$(./build/bin/llama-bench -m "$MODEL" -ngl 99 -fa 1 -ctk "$K" -ctv "$V" -t 10 -p 512 -n 128 -r 1 2>/dev/null)
    PP=$(echo "$BENCH" | grep "pp" | sed 's/.*| *\([0-9.]*\) ±.*/\1/')
    TG=$(echo "$BENCH" | grep "tg" | sed 's/.*| *\([0-9.]*\) ±.*/\1/')
    echo "  $K + $V: pp=$PP tg=$TG"
    # Update CSV row with speed
    sed -i '' "s/^\($MNAME,$K,$V,[^,]*,[^,]*,[^,]*\),,$/\1,$PP,$TG/" "$CSV"
done

# --- Sinks ---
SCSV="$OUTDIR/benchmark-${MNAME}-sinks.csv"
echo "model,k_type,v_type,sink,calib_tokens,ppl" > "$SCSV"

echo ""
echo "========== 5hi_3lo_had calibration windows (${CHUNKS} chunks, c=${CTX}) =========="
for V in tqv_had_mse4 q4_0; do
    echo "--- V = $V ---"
    for SINK in 32 64 128 256; do
        echo -n "  sink=$SINK: "
        OUT=$(./build/bin/llama-perplexity -m "$MODEL" -f "$WIKI" -ctk tqk_5hi_3lo_had -ctv "$V" --tq-sinks $SINK -ngl 99 --flash-attn on --chunks $CHUNKS -c $CTX 2>&1)
        PPL=$(echo "$OUT" | grep "Final" | sed 's/.*PPL = \([0-9.]*\).*/\1/')
        CALIB=$(echo "$OUT" | grep "accumulated" | sed 's/.*\([0-9]*\) tokens accumulated.*/\1/' | head -1)
        echo "PPL = $PPL (calib=$CALIB)"
        echo "$MNAME,tqk_5hi_3lo_had,$V,$SINK,$CALIB,$PPL" >> "$SCSV"
    done
done

# --- Long context ---
LCSV="$OUTDIR/benchmark-${MNAME}-16k.csv"
echo "model,k_type,v_type,sink,ppl" > "$LCSV"

echo ""
echo "========== LONG CONTEXT: 1 chunk × 16K =========="
for COMBO in "f16:f16:0" "q8_0:q8_0:0" "q4_0:q4_0:0" "tqk_had_mse4:tqv_had_mse4:0" "tqk_5hi_3lo_had:tqv_had_mse4:0" "tqk_5hi_3lo_had:tqv_had_mse4:64" "tqk_5hi_3lo_had:tqv_had_mse4:128" "tqk_5hi_3lo_had:tqv_had_mse4:256"; do
    K=$(echo $COMBO | cut -d: -f1)
    V=$(echo $COMBO | cut -d: -f2)
    SINK=$(echo $COMBO | cut -d: -f3)
    SINK_FLAG=""
    if [ "$SINK" != "0" ]; then
        SINK_FLAG="--tq-sinks $SINK"
    fi
    echo -n "  $K + $V (sink=$SINK): "
    PPL=$(./build/bin/llama-perplexity -m "$MODEL" -f "$WIKI" -ctk "$K" -ctv "$V" $SINK_FLAG -ngl 99 --flash-attn on --chunks 1 -c 16384 2>&1 | grep "Final" | sed 's/.*PPL = \([0-9.]*\).*/\1/')
    echo "PPL = $PPL"
    echo "$MNAME,$K,$V,$SINK,$PPL" >> "$LCSV"
done

echo ""
echo "Results saved to:"
echo "  $CSV"
echo "  $SCSV"
echo "  $LCSV"
