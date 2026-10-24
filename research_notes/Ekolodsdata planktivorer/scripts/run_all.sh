#!/bin/bash
# Reproduce every table from the downloaded ICES files. Run from the note folder:
#   bash scripts/run_all.sh            (downloads are skipped when present)
set -e
cd "$(dirname "$0")/.."
python3 -I scripts/list_submissions.py data/ices_acoustic_uploadlist.json > tables/ices_acoustic_submissions.tsv
DK=$(awk -F'\t' '$2=="HERAS" && $3=="DK"{print "data/ices_zip/acoustic_"$1".zip"}' tables/ices_acoustic_submissions.tsv)
python3 -I scripts/acoustic_box.py data/derived/nasc_region_logs.tsv $DK 2> tables/acoustic_cruise_summary.txt
python3 -I scripts/biotic_region.py data/derived/biotic_region data/ices_zip/biotic_*.zip
rm -f data/derived/biotic_region_lengths.tsv
A="data/derived/nasc_region_logs.tsv data/derived/biotic_region_hauls.tsv data/derived/biotic_region_catch.tsv data/derived/biotic_region_individuals.tsv tables"
python3 -I scripts/estimate_box.py $A --boot 1000
python3 -I scripts/estimate_box.py $A --boot 50 --fot-only --tag _fot
python3 -I scripts/estimate_box.py $A --boot 50 --mackerel --tag _withmac
python3 -I scripts/estimate_box.py $A --boot 50 --hauls-in-box --tag _haulsinbox
python3 -I scripts/summarise.py tables/acoustic_box_by_year.csv tables/acoustic_box_summary.csv
python3 -I scripts/nasc_rectangles.py data/derived/nasc_region_logs.tsv tables
python3 -I scripts/iiia_check.py $A
python3 -I scripts/nasc_depth_profile.py tables/nasc_depth_profile_box.csv $DK
python3 -I scripts/iessns_swept.py tables/iessns_dk_skagerrak_swept.csv 65 $(awk -F'\t' '$2=="IESSNS" && $3=="DK"{print "data/ices_zip/biotic_"$1".zip"}' tables/ices_acoustic_submissions.tsv)
