#!/bin/bash
# usage: ./append_tag.sh <tag_value> [directory]

tag_value="$1"
dir="${2:-.}"

if [ -z "$tag_value" ]; then
    echo "Usage: $0 <tag_value> [directory]"
    exit 1
fi

for f in "$dir"/*.txt; do
    echo "tag: $tag_value" >> "$f"
    echo "Updated: $f"
done
