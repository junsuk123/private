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

The first command only issues an access token. The second command also calls
the read-only balance endpoint and never places orders. KIS can reject repeated
token issuance for about one minute, so wait briefly before retrying.

PowerShell load example:

```powershell
Get-Content config/secrets/kis_api_keys.env |
  Where-Object { $_ -and -not $_.StartsWith("#") } |
  ForEach-Object {
    $name, $value = $_ -split "=", 2
    [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
  }
```
