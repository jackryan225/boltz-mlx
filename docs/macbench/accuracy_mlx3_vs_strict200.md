# Accuracy vs `strict200` (PREREG_MACBENCH.md section 5)

tags used: mlx-3-20261031a


## f16_50: NO MATCH or incomplete: 9b3h_empty=FAIL

| input | status | S tm (dissim / limit) | S lddt (dissim / limit) | E tm (cand / limit) | E lddt (cand / limit) | C failing keys |
|---|---|---|---|---|---|---|
| 9b3h_empty | FAIL | 0.6596 / 0.7758 | 0.6100 / 0.7632 | 0.354 / 0.317 | 0.421 / 0.398 | iptm |
| 9b3h_msa | PASS | 0.2148 / 0.3432 | 0.2565 / 0.3331 | 0.704 / 0.681 | 0.815 / 0.789 | - |
| 9moj_empty | PASS | 0.7329 / 0.9028 | 0.5721 / 0.7403 | 0.193 / 0.136 | 0.365 / 0.321 | - |
| 9moj_msa | PASS | 0.0074 / 0.0171 | 0.0267 / 0.0411 | 0.959 / 0.940 | 0.922 / 0.899 | - |

## screen: MATCH (every input, every rule family)

| input | status | S tm (dissim / limit) | S lddt (dissim / limit) | E tm (cand / limit) | E lddt (cand / limit) | C failing keys |
|---|---|---|---|---|---|---|
| 9b3h_empty | PASS | 0.6265 / 0.7758 | 0.5958 / 0.7632 | 0.350 / 0.317 | 0.427 / 0.398 | - |
| 9b3h_msa | PASS | 0.2359 / 0.3432 | 0.2549 / 0.3331 | 0.706 / 0.681 | 0.816 / 0.789 | - |
| 9moj_empty | PASS | 0.7153 / 0.9028 | 0.5883 / 0.7403 | 0.200 / 0.136 | 0.380 / 0.321 | - |
| 9moj_msa | PASS | 0.0090 / 0.0171 | 0.0270 / 0.0411 | 0.968 / 0.940 | 0.927 / 0.899 | - |

## f16_200: MATCH (every input, every rule family)

| input | status | S tm (dissim / limit) | S lddt (dissim / limit) | E tm (cand / limit) | E lddt (cand / limit) | C failing keys |
|---|---|---|---|---|---|---|
| 9b3h_empty | PASS | 0.4271 / 0.7758 | 0.4690 / 0.7632 | 0.361 / 0.317 | 0.419 / 0.398 | - |
| 9b3h_msa | PASS | 0.1832 / 0.3432 | 0.1807 / 0.3331 | 0.701 / 0.681 | 0.810 / 0.789 | - |
| 9moj_empty | PASS | 0.5271 / 0.9028 | 0.4861 / 0.7403 | 0.176 / 0.136 | 0.357 / 0.321 | - |
| 9moj_msa | PASS | 0.0047 / 0.0171 | 0.0204 / 0.0411 | 0.960 / 0.940 | 0.918 / 0.899 | - |
