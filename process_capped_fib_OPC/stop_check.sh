#!/bin/bash
# stop_check.sh -- early-stopping check for the capped-fibril production run.
#
# Run it at fixed checkpoints only: 50, 75, 100, 125 ... ns, while mdrun keeps going:
#     bash stop_check.sh
#
# A checkpoint PASSES when every line of check_capped_stability_3.py's verdict
# passes (the cap junction K-L has settled, sits below the H-bond ceiling and
# behaves like the free end A-B; the interior has settled) AND the fibril has
# never come within 2.0 nm of its own periodic image (the run is unrestrained).
# STOP when two checkpoints in a row pass -- so 75 ns at the earliest.
#
# Checking at fixed points, and needing two passes in a row, is what keeps this
# honest: test often enough and a drifting run will pass once by luck.
#
# Before the first use, set at the top of check_capped_stability_3.py:
#     HETS_FIRST = 217   HETS_LEN = 73   RESTRAINED = ()   STRIDE = 1
# (STRIDE = 1 because the trajectory below is already thinned to one frame per
# 50 ps, the same spacing STRIDE = 5 gives on the 10 ps original.)
set -u
GMX=gmx_mpi
MIN_IMAGE=2.0     # nm: 2 x rvdw

now=$(grep -A1 "Step  *Time" prod.log | tail -n 1 | awk '{printf "%.1f", $2/1000}')
[ -n "$now" ] || { echo "cannot read the current time from prod.log"; exit 1; }
echo "run is at $now ns"

# protein-only files: 10x smaller than the full system, and all F8 needs
if [ ! -f prot.tpr ]; then
    printf "q\n"       | $GMX make_ndx -f prod.tpr -o default.ndx >/dev/null 2>&1
    printf "Protein\n" | $GMX convert-tpr -s prod.tpr -n default.ndx -o prot.tpr >/dev/null 2>&1
fi
printf "Protein\n" | $GMX trjconv -s prod.tpr -f prod.cpt -o prot_so_far.gro >/dev/null 2>&1
printf "Protein\n" | $GMX trjconv -s prod.tpr -f prod.xtc -o nojump_prot.xtc \
                        -pbc nojump -dt 50 >/dev/null 2>&1 \
    || { echo "trjconv failed"; exit 1; }

out=check_${now}ns.txt
python3 check_capped_stability_3.py prot_so_far.gro nojump_prot.xtc > "$out" 2>&1
if ! grep -q "verdict" "$out"; then
    echo "check script did not finish -- see $out"; tail -n 5 "$out"; exit 1
fi
nfail=$(sed -n '/^ *verdict *$/,$p' "$out" | grep -c "\[FAIL\]")

# closest approach to its own image, one frame per ns over the whole run so far
printf "Protein\n" | $GMX mindist -s prot.tpr -f nojump_prot.xtc -pi -dt 1000 \
                        -od mindist_pi.xvg >/dev/null 2>&1
minpi=$(awk '!/^[#@]/ {if (m=="" || $2<m) m=$2} END {printf "%.2f", m}' mindist_pi.xvg)

cap=$(grep -m1 "cap junction" "$out" | sed 's/^ *//')
end=$(grep -m1 "other free end (" "$out" | sed 's/^ *//')
img_ok=$(awk -v m="$minpi" -v t="$MIN_IMAGE" 'BEGIN {print (m != "" && m >= t) ? 1 : 0}')
if [ "$nfail" -eq 0 ] && [ "$img_ok" -eq 1 ]; then verdict=PASS; else verdict=FAIL; fi

prev=$(tail -n 1 stop_log.txt 2>/dev/null | awk '{print $3}')    # "<t> ns PASS|FAIL ..."
printf "%7s ns  %s  fails=%d  min_image=%s nm  | %s | %s\n" \
    "$now" "$verdict" "$nfail" "$minpi" "$cap" "$end" >> stop_log.txt

echo
sed -n '/^ *verdict *$/,/^$/p' "$out"
echo "closest approach to its own image so far: $minpi nm (needs >= $MIN_IMAGE)"
echo
echo "checkpoint log:"; cat stop_log.txt
echo
if [ "$verdict" = PASS ] && [ "$prev" = PASS ]; then
    echo "STOP: two checkpoints in a row pass. End the run cleanly with"
    echo "    pkill -TERM -f 'gmx_mpi mdrun'"
    echo "and take the final structure from prod.cpt."
elif [ "$verdict" = PASS ]; then
    echo "PASS, but the previous checkpoint did not -- run on to the next checkpoint."
else
    echo "FAIL -- keep running; full output in $out."
fi
