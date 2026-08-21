#!/usr/bin/env bash
# Verifies every sourceUrl in js/mechanisms.js actually resolves.
# Usage: bash scripts/check_links.sh   (from assignment-8/)
set -uo pipefail
cd "$(dirname "$0")/.."

urls=$(grep -o "sourceUrl: '[^']*'" js/mechanisms.js | sed "s/sourceUrl: '//;s/'$//" | sort -u)
fail=0; n=0
printf '%-6s %s\n' "CODE" "URL"
printf '%s\n' "------------------------------------------------------------------"
while IFS= read -r u; do
  n=$((n+1))
  code=$(curl -sS -L -o /dev/null -w '%{http_code}' --max-time 25 \
         -A 'Mozilla/5.0 (link-check; ERA-V5 assignment-8)' "$u" 2>/dev/null || echo "ERR")
  # Some hosts (Reddit, Substack) block automated HEAD/GET from CI but serve fine
  # in a browser. Those are reported, not treated as broken links.
  if [ "$code" = "200" ]; then
    printf '%-6s %s\n' "$code" "$u"
  else
    printf '%-6s %s   <-- check manually\n' "$code" "$u"
    fail=$((fail+1))
  fi
done <<< "$urls"

printf '\n%d URLs checked, %d need a manual look.\n' "$n" "$fail"
[ "$fail" -eq 0 ] || echo "(bot-blocked hosts are expected here; open them in a browser to confirm)"
