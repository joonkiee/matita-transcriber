; Inno Setup script for Matita on Windows. Built by scripts\build-windows.ps1:
;   ISCC /DAppVersion=3.22.0 /DAppDir=dist\Matita /DOutDir=dist scripts\matita.iss
; Installs for the current user only (no administrator password), into %LOCALAPPDATA%\Programs\Matita.
; Transcripts (Documents\Transcripts), settings and models (%LOCALAPPDATA%\Matita) are never touched.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6F1B8E2A-4C1D-4E7B-9A55-3D2C8B9F7A10}
AppName=Matita
AppVersion={#AppVersion}
AppVerName=Matita {#AppVersion}
AppPublisher=Matita (open source)
AppPublisherURL=https://github.com/joonkiee/matita-transcriber
AppSupportURL=https://github.com/joonkiee/matita-transcriber/issues
AppUpdatesURL=https://github.com/joonkiee/matita-transcriber/releases
DefaultDirName={localappdata}\Programs\Matita
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutDir}
OutputBaseFilename=Matita-{#AppVersion}-setup
SetupIconFile={#AppDir}\matita.ico
UninstallDisplayIcon={app}\matita.ico
UninstallDisplayName=Matita
LicenseFile={#AppDir}\LICENSE.txt
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ko"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#AppDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; files from an older version that the new one doesn't ship
Type: filesandordirs; Name: "{app}\bin"
Type: filesandordirs; Name: "{app}\python"

[Icons]
Name: "{userprograms}\Matita"; Filename: "{app}\python\pythonw.exe"; Parameters: "-X utf8 ""{app}\matita.py"""; WorkingDir: "{app}"; IconFilename: "{app}\matita.ico"; Comment: "Transcripts, summaries and notes from your recordings"
Name: "{userdesktop}\Matita"; Filename: "{app}\python\pythonw.exe"; Parameters: "-X utf8 ""{app}\matita.py"""; WorkingDir: "{app}"; IconFilename: "{app}\matita.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-X utf8 ""{app}\matita.py"""; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,Matita}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; stop Matita if it's running, so its files can be removed
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -NonInteractive -Command ""Get-CimInstance Win32_Process | Where-Object {{ $_.ExecutablePath -like '{app}\python\*' } | Invoke-CimMethod -MethodName Terminate"""; Flags: runhidden; RunOnceId: "StopMatita"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
