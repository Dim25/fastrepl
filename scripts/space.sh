#!/bin/bash

# NOTE: [-i '' -e] is Mac specific. For other system, [-i] will work
sed -i '' -e '/datasets:/ {
        :a
        N
        /tags:/!ba
        s|datasets:.*\ntags:|emoji: ⚡\ncolorFrom: blue\ncolorTo: red\ntags:|
}' spaces/$1/README.md
