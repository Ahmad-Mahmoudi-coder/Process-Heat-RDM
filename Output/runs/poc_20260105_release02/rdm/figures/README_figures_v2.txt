RDM Figures V2 - Improvements and Changes
================================================

This folder contains improved v2 versions of RDM figures with the following enhancements:

1. LABEL MAPPINGS
   - All strategy and upgrade labels have been standardized to thesis-friendly names
   - Strategy labels: "AUTO (min total cost)", "NO UPGRADE (shedding allowed)", etc.
   - Upgrade labels: "GXP upgrade opt1 (+21 MW)", "GXP upgrade opt2 (+32 MW, N-1)", etc.

2. TITLE CLEANUP
   - Removed bundle names (poc_20260105_release02) from all figure titles
   - Titles now use format: "2035_EB | Description" or "2035_BB | Description"

3. NEW V2 FIGURES

   V2A: AUTO-only upgrade frequency
   - Simple bar charts showing upgrade choice frequency under AUTO strategy only
   - More interpretable than forced-strategy stacked bars
   - Files: grid_rdm_2035_EB__auto_upgrade_frequency_v2.png
            grid_rdm_2035_BB__auto_upgrade_frequency_v2.png
            grid_rdm_2035_EB_vs_BB__auto_upgrade_frequency_v2.png (combined)

   V2B: Strategy × upgrade frequency heatmap
   - Heatmap showing percentage of futures for each strategy×upgrade combination
   - All strategies included (AUTO + forced + no-upgrade)
   - Annotated with percentages for readability
   - Files: grid_rdm_2035_EB__strategy_upgrade_heatmap_v2.png
            grid_rdm_2035_BB__strategy_upgrade_heatmap_v2.png

   V2C: AUTO upgrade vs headroom driver
   - Scatter plot showing relationship between headroom multiplier and selected upgrade
   - Points jittered slightly for better visibility
   - Files: grid_rdm_2035_EB__auto_upgrade_vs_headroom_v2.png
            grid_rdm_2035_BB__auto_upgrade_vs_headroom_v2.png

4. PRESERVATION OF ORIGINAL FIGURES
   - All original figures remain untouched
   - V2 figures use "_v2" suffix to avoid overwriting
   - Original figures can still be found without the "_v2" suffix

WHY THESE CHANGES?
- AUTO-only frequency charts are more interpretable for policy analysis
- Heatmaps provide clearer visualization of strategy×upgrade relationships
- Cleaner titles improve readability in thesis documents
- Standardized labels ensure consistency across all figures
