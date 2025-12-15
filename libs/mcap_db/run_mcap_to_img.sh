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


PREFIX=
if [ $# == 2 ]; then
    INPUT_ROOT=$1
    OUTPUT_ROOT=$2
elif [ $# == 3 ]; then
    INPUT_ROOT=$1
    OUTPUT_ROOT=$2
    PREFIX=$3
else
    echo "./run_mcap_to_image.sh input_dir output_dir [prefix]"
    exit
fi

TOPIC="/front/zed_node/left/image_rect_color"
TOPIC="/front/zed_node/stereo/image_rect_color"
TOPIC="/ik_trainer/depth_top"
INTERVAL=1

# Check if the provided argument is a directory
if [ ! -d "${INPUT_ROOT}" ]; then
    echo "Error: '${INPUT_ROOT}' is not a valid directory."
    exit 1
fi

# Get the subdirectories
get_subdirectories "${INPUT_ROOT}"

for dir in "${subdirectories[@]}"; do
    echo "$dir"
done

echo "Subdirectories in ${INPUT_ROOT}:${subdirectories[@]}"

starts_with() {
  local string="$1"
  local prefix="$2"

  if [[ -z "$prefix" ]] || [[ "$string" == "$prefix"* ]]; then
    return 0  # true
  else
    return 1  # false
  fi
}


# Get the disk usage for each subdirectory
for bag_name in "${subdirectories[@]}"; do
	echo "Bag name: ${bag_name}"
	if starts_with "${bag_name}" "${PREFIX}"; then
		if [[ $(realpath "${INPUT_ROOT}/${bag_name}") != $(realpath "${INPUT_ROOT}") ]]; then
			echo "Convert mcap to images for ${INPUT_ROOT}/${bag_name}:"
			python3 mcap_to_img.py \
				-f ${INPUT_ROOT} \
				-t ${TOPIC} \
				-p ${bag_name} \
				-b ${bag_name} -i ${INTERVAL} -o ${OUTPUT_ROOT}/${bag_name}
		fi
	fi
done | sort -h



