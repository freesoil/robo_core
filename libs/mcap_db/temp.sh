#!/bin/bash

# Function to get the subdirectory names
get_subdirectories() {
    local path=$1
    # Resolve the path if it is a symlink
    if [ -L "$path" ]; then
        path=$(readlink -f "$path")
    fi
    # Remove trailing slash if it exists
    path="${path%/}"
    # Find all subdirectories in the specified path and store them in an array
    subdirectories=()
    while IFS= read -r -d '' dir; do
        subdirectories+=("$(basename "$dir")")
    done < <(find "$path" -mindepth 1 -maxdepth 1 -type d -print0)
}

# Usage
path_to_search=$1

# Check if the path is provided
if [ -z "$path_to_search" ]; then
    echo "Please provide a path to search."
    exit 1
fi

# Get the subdirectories
get_subdirectories "$path_to_search"

# Print the subdirectories
echo "Subdirectories in '$path_to_search':"
for dir in "${subdirectories[@]}"; do
    echo "$dir"
done

