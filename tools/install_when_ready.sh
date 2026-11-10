#!/bin/zsh
# Waits for the iPhone, then builds Release, installs and launches FOCUS.
D=A3D6F69C-8877-56F3-8B2E-20EDC8D0B7DB; ID=00008120-001914AC08834032
cd "$(dirname $0)/.."
for i in $(seq 1 120); do
  xcrun devicectl list devices 2>&1 | grep -q "$D.*connected" && break; sleep 5
done
xcodebuild -project Focus.xcodeproj -scheme Focus -configuration Release -destination "id=$ID" -allowProvisioningUpdates build -quiet 2>&1 | grep -E "error:" | head -3
APP=$(find ~/Library/Developer/Xcode/DerivedData/Focus-*/Build/Products/Release-iphoneos -name Focus.app -maxdepth 1 | head -1)
xcrun devicectl device install app --device $D "$APP" 2>&1 | grep -iE "installed|error"
xcrun devicectl device process launch --device $D --terminate-existing com.tohraan.focus-dr 2>&1 | tail -1
