#!/bin/sh
# Usage: sh pdftxt.sh <pdf>  -> writes <pdf>.txt next to it, only if it is a PDF
if file "$1" | grep -q 'PDF document'; then pdftotext -layout "$1" "$1.txt" 2>/dev/null && wc -w "$1.txt"; else echo "NOT A PDF: $(file -b "$1" | cut -c1-60)"; fi
