# usage: awk -v from=50 -f blocks.awk free_eq.dat
!/^#/ {
  b = int($1 / 10000); n[b]++; a[b] += $2; r[b] += $3; c[b] += $4
  if ($1 > last) last = $1
  if ($1 >= from * 1000) {
    m++; x += $2; y += $4; xx += $2 * $2; yy += $4 * $4; xy += $2 * $4
    q = int(($1 - from * 1000) / 25000); nq[q]++; aq[q] += $2; cq[q] += $4
  }
}
END {
  for (k = 0; k in n; k++) if (n[k] > 1)
    printf "%3d-%3d ns  adsorbed %5.2f  Rg %5.2f  cluster %5.2f\n", k*10, k*10+10, a[k]/n[k], r[k]/n[k], c[k]/n[k]
  printf "\n25 ns blocks from %d ns:\n", from
  for (k = 0; k in nq; k++) if (nq[k] > 1 && from*1000 + (k+1)*25000 <= last + 1) {
    A = aq[k]/nq[k]; C = cq[k]/nq[k]; nb++; sa += A; sc += C; saa += A*A; scc += C*C
    printf "  %3d-%3d ns  adsorbed %5.2f  cluster %5.2f\n", from + k*25, from + k*25 + 25, A, C
  }
  if (nb > 1) {
    ma = sa/nb; mc = sc/nb
    printf "  mean of blocks: adsorbed %.2f +/- %.2f, cluster %.2f +/- %.2f  (standard error, %d blocks)\n",
      ma, sqrt((saa/nb - ma*ma) / (nb-1)), mc, sqrt((scc/nb - mc*mc) / (nb-1)), nb
  }
  rr = (m*xy - x*y) / sqrt((m*xx - x*x) * (m*yy - y*y))
  printf "\ncorrelation of adsorbed with largest cluster from %d ns: %.2f\n", from, rr
}
