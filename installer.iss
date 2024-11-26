[Setup]
AppName=VoiceAuth
AppVersion=1.0
DefaultDirName={autopf}\VoiceAuth
DefaultGroupName=VoiceAuth
OutputDir=dist\installer
OutputBaseFilename=VoiceAuthInstaller

[Files]
Source: "dist\VoiceAuth.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "images\splash.jpg"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\VoiceAuth"; Filename: "{app}\VoiceAuth.exe"

[Run]
Filename: "{app}\VoiceAuth.exe"; Description: "Launch VoiceAuth"; Flags: nowait postinstall skipifsilent

[Messages]
SplashScreen="images\splash.jpg"
