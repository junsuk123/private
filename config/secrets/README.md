# Local Secrets

Put real credentials in files under this directory. These files are ignored by
Git, except for `*.example` templates and this README.

For Korea Investment & Securities Open API, copy:

```powershell
Copy-Item config/secrets/kis_api_keys.env.example config/secrets/kis_api_keys.env
```

Then fill in `config/secrets/kis_api_keys.env` locally.

`KisDevelopersApiClient` loads this file automatically without printing values.
Environment variables that are already set are preserved unless the loader is
called with `override=True`.

Important toggles:

- `KIS_PAPER_TRADING=true` uses the KIS virtual domain.
- `KIS_PAPER_TRADING=false` selects the live KIS domain.
- `KIS_LIVE_ENABLED=false` blocks KIS order, status, and account calls.
- `KIS_LIVE_ENABLED=true` should stay off until the manual approval and risk
  gates are intentionally enabled.

Read-only connection checks:

```powershell
python scripts/check_kis_connection.py
python scripts/check_kis_connection.py --account
```

The first command makes an access token available. If a valid cached token
exists, it reuses that token instead of requesting a new one. The second command
also calls the read-only balance endpoint and never places orders.

Access tokens are cached under this ignored directory:

- `config/secrets/kis_access_token.paper.json`
- `config/secrets/kis_access_token.live.json`

Live readiness checks use this order:

1. A manually supplied `KIS_LIVE_ACCESS_TOKEN` or `KIS_ACCESS_TOKEN`
2. The live token cache file
3. A new `/oauth2/tokenP` request, only when no valid token exists

The web server runs a read-only live-readiness account check automatically on
startup when `AUTO_START_LIVE_READINESS=true` (the default). That startup check
uses the same token reuse order above and only reads balance/holdings state; it
does not place orders.

If KIS already issued today's token outside this app and the cache file is not
present, paste that token into `KIS_LIVE_ACCESS_TOKEN` in
`config/secrets/kis_api_keys.env`, then run live readiness again. The app will
use that token for the read-only live account check without issuing a new token.

The token cache is treated like a secret and must not be committed or shared.
KIS access tokens are valid for about 24 hours, and repeated token issuance can
be limited, so keep using the cached token until it expires. To intentionally
force a new token after expiry or revocation, delete the matching cache file and
run the check again.

Before requesting a new token, the client verifies that the cache path is
writable. After a token is issued, it writes the cache atomically and reads it
back to confirm the same token was saved.

PowerShell load example:

```powershell
Get-Content config/secrets/kis_api_keys.env |
  Where-Object { $_ -and -not $_.StartsWith("#") } |
  ForEach-Object {
    $name, $value = $_ -split "=", 2
    [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
  }
```
