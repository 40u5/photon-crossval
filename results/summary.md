| scenes | photons/px | PSNR (dB) | GT: E[(err/sd)^2] | photon E[z^2] | flagged windows | 90% coverage raw -> recal. | AUROC anomaly: std / photon z / recal. std |
|---|---|---|---|---|---|---|---|
| clean | 2 | 18.0 | 14.83 | 1.58 | 0.4% | 0.34 -> 0.77 | - |
| anomaly | 2 | 17.5 | 13.66 | 1.63 | 0.5% | 0.35 -> 0.78 | 0.57 / 0.48 / 0.54 |
| clean | 8 | 25.7 | 7.70 | 1.15 | 0.4% | 0.73 -> 0.87 | - |
| anomaly | 8 | 24.9 | 6.80 | 1.27 | 0.4% | 0.72 -> 0.90 | 0.69 / 0.49 / 0.49 |
| clean | 32 | 29.5 | 12.76 | 1.19 | 0.6% | 0.75 -> 0.89 | - |
| anomaly | 32 | 29.0 | 12.29 | 1.23 | 0.1% | 0.78 -> 0.91 | 0.60 / 0.54 / 0.62 |

- lowres: photon_z2=0.96, flag_rate=0.00, psnr=23.16, err_in_anomaly=0.29
- mono: photon_z2=0.83, flag_rate=0.01, psnr=14.47, err_in_anomaly=0.31
