# Archived prediction products

`continuous_rey_paper_summary_2d.nc` is the paper-facing 2D summary of the
formal seed-714 prediction batch. It has a 1200 × 1701 regular latitude–longitude
grid at 0.1° spacing and summarizes 27 age layers from 0 to 2580 ka. Longitude
is stored from 120°E to 290°E; values greater than 180° correspond to western
longitudes when expressed in the −180° to 180° convention.

## Variable dictionary

| Variable | Meaning |
|---|---|
| `max_pred_rey_ppm_0_2580ka` | Maximum predicted REY concentration across age layers (ppm). |
| `p95_pred_rey_ppm_0_2580ka` | 95th percentile of predicted REY across age layers (ppm). |
| `mean_pred_rey_ppm_0_2580ka` | Mean predicted REY across age layers (ppm). |
| `max_prob_rey_ge_1000ppm_0_2580ka` | Maximum predicted P(REY ≥ 1000 ppm) across age layers. |
| `max_prob_rey_ge_1500ppm_0_2580ka` | Maximum predicted P(REY ≥ 1500 ppm) across age layers. |
| `is_prob_rey_ge_1000ppm_0_2580ka` | Binary mask: maximum P(REY ≥ 1000 ppm) is at least 0.5. |
| `is_prob_rey_ge_1500ppm_0_2580ka` | Binary mask: maximum P(REY ≥ 1500 ppm) is at least 0.5. |
| `is_peak_pred_rey_ge_1000ppm_0_2580ka` | Binary mask: peak continuous REY prediction is at least 1000 ppm. |
| `is_peak_pred_rey_ge_1500ppm_0_2580ka` | Binary mask: peak continuous REY prediction is at least 1500 ppm. |
| `high_value_tier_id_0_2580ka` | Probability tier: 0 = neither probability reaches 0.5; 1 = P1000 reaches 0.5 but P1500 does not; 2 = P1500 reaches 0.5. |
| `peak_age_ka_pred_rey` | Age layer of maximum continuous predicted REY (ka). |
| `peak_age_ka_prob_ge_1500ppm` | Age layer of maximum P(REY ≥ 1500 ppm) (ka). |
| `dominant_mechanism_class_id_0_2580ka` | Dominant mean mechanism class: 0 carbonate-diluted background, 1 carrier-rich enrichment, 2 low-dilution pelagic accumulation. |
| `dominant_mechanism_confidence_0_2580ka` | Mean probability of the dominant mechanism class. |
| `prob_carbonate_background_mean_0_2580ka` | Mean probability of carbonate-diluted background across age layers. |
| `prob_class1_enriched_mean_0_2580ka` | Mean probability of carrier-rich enrichment across age layers. |
| `prob_class2_slow_sedimentation_mean_0_2580ka` | Mean probability of low-dilution pelagic accumulation across age layers. |
| `prediction_confidence_mean_0_2580ka` | Mean prediction-confidence field across age layers. |
| `screening_score_ge1000` | Maximum P(REY ≥ 1000 ppm) multiplied by mean prediction confidence. |
| `screening_score_ge1500` | Maximum P(REY ≥ 1500 ppm) multiplied by mean prediction confidence. |

Probabilities and scores are dimensionless in the interval [0, 1]. Binary masks
use 0 and 1. The NetCDF contains unmasked numerical model output; geological
interpretation should apply the reliability/confidence layers and the water-depth
criteria described in the manuscript.

## Age-resolved 3D product

The 27-layer archive is distributed as three NetCDF files on the same
1200 × 1701 grid. All coordinates and variable values match the original product:

| File | Age layers (ka) |
|---|---|
| `rey_predictions_3d_part01.nc` | 0, 100, ..., 800 |
| `rey_predictions_3d_part02.nc` | 900, 1000, ..., 1700 |
| `rey_predictions_3d_part03.nc` | 1800, 1900, ..., 2500, 2580 |

Use `tools.open_map_parts.open_maps()` to read the complete age series.
Its variables include physical
depth, continuous `pred_logrey` and `pred_rey_ppm`, predictive uncertainty and
confidence, three mechanism-facies probabilities, ordered probabilities for
600/800/1000/1500 ppm, threshold masks and high-value tiers.

`continuous_rey_prediction_summary.json` records the formal batch settings.
Rendered PNG maps are not included; they can be regenerated from the archived
3D and 2D numerical products.
