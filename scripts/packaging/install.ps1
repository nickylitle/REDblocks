# Simblocks installer for one Windows user. No administrator rights needed.
#   Installs the folder build (dist\Simblocks) to %LOCALAPPDATA%\Programs\Simblocks and adds:
#     - a Start Menu entry, so typing "Simblocks" in Windows search finds it (right-click > Pin to taskbar)
#     - an optional desktop shortcut
#     - "Simblocks" in Windows Run (Win+R) via App Paths
#     - Simblocks in the "Open with" list for .yaml / .yml model files
#     - an entry in Settings > Apps, so it uninstalls cleanly
#   Your settings, saved groups and models are never touched (they live in %USERPROFILE%\.simblocks and your folders).
#
# Normally started by build_exes.bat. By hand:
#   powershell -NoProfile -ExecutionPolicy Bypass -File packaging\install.ps1            install / update
#   powershell -NoProfile -ExecutionPolicy Bypass -File packaging\install.ps1 -Uninstall remove
param([switch]$Uninstall, [switch]$Quiet)
$ErrorActionPreference = 'Stop'

$Dest     = Join-Path $env:LOCALAPPDATA 'Programs\Simblocks'
$Exe      = Join-Path $Dest 'Simblocks.exe'
$StartLnk = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Simblocks.lnk'
$DeskDir  = [Environment]::GetFolderPath('Desktop')
if (-not $DeskDir) { $DeskDir = Join-Path $HOME 'Desktop' }
$DeskLnk  = Join-Path $DeskDir 'Simblocks.lnk'
$UninstKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Simblocks'
$AppPath   = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\App Paths\Simblocks.exe'
$ProgId    = 'Simblocks.Model'
$Classes   = 'HKCU:\Software\Classes'

function Stop-Simblocks {
    Get-Process -Name 'Simblocks' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}

function Remove-Registration {
    foreach ($k in @($UninstKey, $AppPath, "$Classes\$ProgId")) {
        if (Test-Path $k) { Remove-Item $k -Recurse -Force }
    }
    foreach ($ext in @('.yaml', '.yml')) {
        $owp = "$Classes\$ext\OpenWithProgids"
        if (Test-Path $owp) { Remove-ItemProperty -Path $owp -Name $ProgId -ErrorAction SilentlyContinue }
    }
    foreach ($l in @($StartLnk, $DeskLnk)) { if (Test-Path $l) { Remove-Item $l -Force } }
}

if ($Uninstall) {
    Stop-Simblocks
    Remove-Registration
    if (Test-Path $Dest) {
        # this script may be running from inside $Dest: remove everything else now, the folder itself just after
        Get-ChildItem $Dest | Where-Object { $_.FullName -ne $PSCommandPath } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Start-Process -WindowStyle Hidden cmd.exe -ArgumentList "/c timeout /t 2 >nul & rmdir /s /q `"$Dest`""
    }
    Write-Host 'Simblocks removed. Your settings, saved groups and models were kept (%USERPROFILE%\.simblocks).'
    exit 0
}

# ------------------------------------------------------------------ install / update
$Src = Join-Path $PSScriptRoot '..\dist\Simblocks'
if (-not (Test-Path (Join-Path $Src 'Simblocks.exe'))) {
    throw "dist\Simblocks\Simblocks.exe not found. Run packaging\build_exes.bat first."
}
Stop-Simblocks
if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
New-Item -ItemType Directory -Path $Dest -Force | Out-Null
Copy-Item -Path (Join-Path $Src '*') -Destination $Dest -Recurse -Force
Copy-Item -Path $PSCommandPath -Destination (Join-Path $Dest 'uninstall.ps1') -Force

# Start Menu shortcut (searchable; right-click it in Start > Pin to taskbar)
$shell = New-Object -ComObject WScript.Shell
function New-Shortcut($path) {
    $s = $shell.CreateShortcut($path)
    $s.TargetPath = $Exe; $s.WorkingDirectory = $Dest; $s.IconLocation = "$Exe,0"
    $s.Description = 'Simblocks: node editor for explicit dynamics sims'
    $s.Save()
}
New-Shortcut $StartLnk
$wantDesk = $false
if (-not $Quiet) { $wantDesk = (Read-Host 'Add a desktop shortcut too? [y/N]') -match '^[yY]' }
if ($wantDesk) { New-Shortcut $DeskLnk }

# Win+R "simblocks"
New-Item -Path $AppPath -Force | Out-Null
Set-Item -Path $AppPath -Value $Exe
Set-ItemProperty -Path $AppPath -Name 'Path' -Value $Dest

# "Open with" for model files (Windows only lets you pick the default yourself: Open with > Choose another app > Always)
New-Item -Path "$Classes\$ProgId\shell\open\command" -Force | Out-Null
Set-Item -Path "$Classes\$ProgId" -Value 'Simblocks model'
Set-Item -Path "$Classes\$ProgId\shell\open\command" -Value "`"$Exe`" `"%1`""
New-Item -Path "$Classes\$ProgId\DefaultIcon" -Force | Out-Null
Set-Item -Path "$Classes\$ProgId\DefaultIcon" -Value "$Exe,0"
foreach ($ext in @('.yaml', '.yml')) {
    New-Item -Path "$Classes\$ext\OpenWithProgids" -Force | Out-Null
    New-ItemProperty -Path "$Classes\$ext\OpenWithProgids" -Name $ProgId -Value '' -PropertyType String -Force | Out-Null
}

# Settings > Apps entry
$sizeKB = [int]((Get-ChildItem $Dest -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1KB)
New-Item -Path $UninstKey -Force | Out-Null
$props = @{
    DisplayName = 'Simblocks'; DisplayVersion = '0.6'; Publisher = 'Simblocks (local build)'
    DisplayIcon = "$Exe,0"; InstallLocation = $Dest
    UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Dest\uninstall.ps1`" -Uninstall"
}
foreach ($k in $props.Keys) { Set-ItemProperty -Path $UninstKey -Name $k -Value $props[$k] }
foreach ($k in @('NoModify', 'NoRepair')) { New-ItemProperty -Path $UninstKey -Name $k -Value 1 -PropertyType DWord -Force | Out-Null }
New-ItemProperty -Path $UninstKey -Name 'EstimatedSize' -Value $sizeKB -PropertyType DWord -Force | Out-Null

Write-Host ''
Write-Host "Installed to $Dest"
Write-Host 'Find it: press the Windows key and type Simblocks. To keep it on the taskbar: right-click it > Pin to taskbar.'
Write-Host 'Open a model by double-clicking it: right-click a .yaml file > Open with > Choose another app > Simblocks > Always.'
Write-Host 'Uninstall any time from Settings > Apps > Simblocks.'
