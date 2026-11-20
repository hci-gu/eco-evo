#!/bin/sh
# Usage: UPW_EMAIL=... sh unpaywall.sh <doi>  -> prints OA locations (email from env only)
curl -s "https://api.unpaywall.org/v2/$1?email=$UPW_EMAIL" | python3 -I -c "
import json,sys
d=json.load(sys.stdin)
print(d.get('title'), d.get('year'), d.get('journal_name'))
print('is_oa', d.get('is_oa'))
for l in d.get('oa_locations') or []:
    print(' ', l.get('url_for_pdf'), '|', l.get('url'))
"
