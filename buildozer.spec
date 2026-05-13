[app]
title = Hyperbolic Geometry
package.name = hyperbolicgeometry
package.domain = com.hyperbolic.geometry

source.dir = .
source.include_exts = py,html,js,css,txt
source.include_patterns = templates/*,static/*
source.exclude_dirs = .github,.git,__pycache__,.buildozer,bin

version = 1.0.0

# flask has a p4a recipe; numpy has a p4a recipe; no plotly needed server-side
requirements = python3,kivy,flask,numpy

android.permissions = INTERNET
android.api = 33
android.minapi = 24
android.ndk = 25b
android.sdk = 33
android.arch = arm64-v8a
android.allow_backup = True
android.copy_libs = 1

orientation = portrait
fullscreen = 0

[buildozer]
log_level = 2
warn_on_root = 1
