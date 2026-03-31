#!/bin/sh
# vim: set ts=4 sw=4 et:
# Download Penn Treebank dataset (for TurboQuant calibration)
#
# Citation:
#   Marcus, Mitchell P., Santorini, Beatrice, and Marcinkiewicz, Mary Ann.
#   "Building a Large Annotated Corpus of English: The Penn Treebank."
#   Computational Linguistics, 19(2), 313-330, 1993.
#   https://www.aclweb.org/anthology/J93-2004

BASE_URL="https://raw.githubusercontent.com/wojzaremba/lstm/master/data"
DIR="ptb"

die() {
    printf "%s\n" "$@" >&2
    exit 1
}

have_cmd() {
    for cmd; do
        command -v "$cmd" >/dev/null || return
    done
}

dl() {
    [ -f "$2" ] && return
    if have_cmd wget; then
        wget "$1" -O "$2"
    elif have_cmd curl; then
        curl -L "$1" -o "$2"
    else
        die "Please install wget or curl"
    fi
}

mkdir -p "$DIR"

dl "$BASE_URL/ptb.train.txt" "$DIR/ptb.train.txt" || exit
dl "$BASE_URL/ptb.valid.txt" "$DIR/ptb.valid.txt" || exit
dl "$BASE_URL/ptb.test.txt"  "$DIR/ptb.test.txt"  || exit

cat <<EOF
Usage (TurboQuant calibration):

  llama-tq-calibrate -m model.gguf -f $DIR/ptb.train.txt -o perms.bin --pre-rope -n 32000

Usage (perplexity evaluation):

  llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw -ctk tqk4_sj --tq-perms perms.bin

EOF
